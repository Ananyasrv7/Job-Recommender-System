from sentence_transformers import SentenceTransformer, util
import pandas as pd

# ── Load model and data ─────────────────────────────────────────────────────
model = SentenceTransformer("esco-sbert")

base = "ESCO dataset - v1.1.1 - classification - en - csv"

skills_df   = pd.read_csv(f"{base}/skills_en.csv")
relations   = pd.read_csv(f"{base}/occupationSkillRelations_en.csv")
occupations = pd.read_csv(f"{base}/occupations_en.csv")


def get_job_skills(job_title, relation_type="essential"):
    """Return required skills for a job title from the ESCO dataset."""
    job = occupations[
        occupations["preferredLabel"].str.lower() == job_title.lower()
    ]
    if job.empty:
        print(f'Job "{job_title}" not found in dataset.')
        return []

    job_id = job.iloc[0]["conceptUri"]

    skill_links = relations[
        (relations["occupationUri"] == job_id) &
        (relations["relationType"] == relation_type)
    ]

    skill_ids = skill_links["skillUri"]
    job_skills = skills_df[skills_df["conceptUri"].isin(skill_ids)]

    return job_skills["preferredLabel"].tolist()


def skill_gap(job_title, user_skills, threshold=0.60):
    """
    Identify missing skills for a target job.
    Batch-encodes all skills at once for speed.
    Only shows missing skills with best_score >= 0.25 to filter irrelevant ones.
    """
    required_skills = get_job_skills(job_title)

    if not required_skills:
        return {"missing": [], "matched": [], "threshold": threshold}

    if not user_skills:
        return {"missing": required_skills, "matched": [], "threshold": threshold}

    # Batch encode all skills at once
    required_embs = model.encode(required_skills, convert_to_tensor=True, show_progress_bar=False)
    user_embs     = model.encode(user_skills,     convert_to_tensor=True, show_progress_bar=False)

    # Similarity matrix: rows = required skills, cols = user skills
    sim_matrix = util.cos_sim(required_embs, user_embs)

    missing = []
    matched = []

    for i, skill in enumerate(required_skills):
        best_score = sim_matrix[i].max().item()
        if best_score >= threshold:
            matched.append((skill, round(best_score, 3)))
        elif best_score >= 0.25:
            # Only show missing skills that are at least somewhat relevant
            missing.append((skill, round(best_score, 3)))
        # Skills scoring below 0.25 are too unrelated to show (e.g. "speak different languages")

    return {
        "missing":   missing,
        "matched":   matched,
        "threshold": threshold,
    }


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    job = input("Enter target job role: ").strip()

    skills_input = input("Enter your skills (comma-separated): ").strip()
    user_skills  = [s.strip() for s in skills_input.split(",") if s.strip()]

    print(f'\nAnalysing skill gap for: "{job}"')
    print(f"Your skills: {user_skills}\n")

    result = skill_gap(job, user_skills)

    print(f"Similarity threshold: {result['threshold']}")

    print(f"\nMatched skills ({len(result['matched'])}):")
    for skill, score in result["matched"]:
        print(f"  +  {skill}  (similarity: {score})")

    print(f"\nMissing skills ({len(result['missing'])}):")
    for skill, score in result["missing"]:
        print(f"  -  {skill}  (best match: {score})")

    if not result["missing"]:
        print("  None — you meet all essential skill requirements!")