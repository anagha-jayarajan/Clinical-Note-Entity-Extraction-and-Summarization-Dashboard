import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_clean_data(csv_path: str = None) -> pd.DataFrame:
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        data = {
            'medical_specialty': [
                'Cardiovascular / Pulmonary', 
                'Gastroenterology', 
                'Neurology', 
                'Cardiovascular / Pulmonary',
                'Orthopedic'
            ],
            'sample_name': [
                'Consult - Chest Pain', 
                'Colonoscopy Report', 
                'Headache Evaluation', 
                'Postop Follow-up',
                'Knee Pain Exam'
            ],
            'transcription': [
                'Patient denies chest pain and shortness of breath. History of severe hypertension and type 2 diabetes. Prescribed metformin and lisinopril.',
                'Patient presents with severe abdominal pain. Denies hematochezia. Previous appendectomy noted. Diagnosed with mild gastritis.',
                'Patient complains of persistent migraine for three days. Denies photophobia or dizziness. Administered sumatriptan.',
                'Follow-up for coronary artery disease. No evidence of myocardial infarction. Patient tolerating aspirin and atorvastatin well.',
                'Patient presents with right knee pain following a fall. No fracture identified on X-ray. Diagnosed with ligament strain and started on ibuprofen.'
            ]
        }
        df = pd.DataFrame(data)

    df = df.dropna(subset=['transcription']).copy()
    df = df.drop_duplicates(subset=['transcription']).copy()
    df['transcription'] = df['transcription'].apply(lambda x: " ".join(x.split()))
    return df

def run_eda(df: pd.DataFrame):
    df['word_count'] = df['transcription'].apply(lambda x: len(x.split()))
    print("=== Note Statistics ===")
    print(df[['medical_specialty', 'word_count']].describe())
    
    plt.figure(figsize=(10, 4))
    sns.countplot(data=df, y='medical_specialty', order=df['medical_specialty'].value_counts().index)
    plt.title('Notes Count by Medical Specialty')
    plt.xlabel('Count')
    plt.ylabel('Specialty')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_and_clean_data()
    run_eda(df)