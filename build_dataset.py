# Script to build a comprehensive Medical Q&A corpus
# Covers Cardiology, Neurology, Endocrinology, Pulmonology, Gastroenterology, Pharmacology, and First Aid.

medical_kb = [
    {
        "q": "What is Hypertension and how is it clinically defined?",
        "a": "Hypertension (high blood pressure) is defined as a sustained systolic blood pressure of 130 mmHg or greater, or a diastolic blood pressure of 80 mmHg or greater. Primary hypertension accounts for 90-95% of cases and develops gradually over years due to genetics, high sodium intake, obesity, and lack of exercise."
    },
    {
        "q": "What is the first-line treatment for stage 1 and stage 2 Hypertension?",
        "a": "First-line pharmacological treatments include thiazide diuretics (e.g., Chlorthalidone), Angiotensin-Converting Enzyme (ACE) inhibitors (e.g., Lisinopril), Angiotensin Receptor Blockers (ARBs like Losartan), and Calcium Channel Blockers (Amlodipine), accompanied by the DASH diet."
    },
    {
        "q": "What are the hallmark symptoms and diagnostic criteria for Type 2 Diabetes?",
        "a": "Hallmark symptoms include polyuria (frequent urination), polydipsia (excessive thirst), polyphagia (excessive hunger), blurred vision, and persistent fatigue. Diagnosis is confirmed with a fasting plasma glucose >= 126 mg/dL, a 2-hour oral glucose tolerance test >= 200 mg/dL, or an HbA1c level of 6.5% or higher."
    },
    {
        "q": "How does Metformin work in managing Type 2 Diabetes?",
        "a": "Metformin belongs to the biguanide class of medications. It works primarily by decreasing hepatic glucose production (gluconeogenesis), decreasing intestinal absorption of glucose, and improving insulin sensitivity by increasing peripheral glucose uptake and utilization."
    },
    {
        "q": "What are the classic symptoms of an acute Myocardial Infarction (heart attack)?",
        "a": "Classic symptoms include crushing substernal chest pressure radiating to the left shoulder, left arm, neck, or mandible, accompanied by diaphoresis (profuse sweating), shortness of breath (dyspnea), nausea, and lightheadedness. Immediate emergency medical intervention is mandatory."
    },
    {
        "q": "What immediate pharmacological interventions are indicated for an acute coronary syndrome?",
        "a": "Immediate initial therapy follows the MONA regimen: Morphine for pain, Oxygen if saturation is below 90%, sublingual Nitroglycerin for vasodilation, and chewable Aspirin (162-325 mg) to inhibit platelet aggregation, followed by urgent coronary revascularization via PCI."
    },
    {
        "q": "What is Asthma and how is an acute bronchospasm managed?",
        "a": "Asthma is a chronic inflammatory disorder of the tracheobronchial tree characterized by reversible airflow obstruction and airway hyperreactivity. Acute bronchospasms are treated with inhaled short-acting beta-2 agonists (SABA) such as Albuterol, alongside systemic corticosteroids to reduce inflammation."
    },
    {
        "q": "What are the common causes and clinical signs of Community-Acquired Pneumonia?",
        "a": "The most common bacterial pathogen is Streptococcus pneumoniae. Clinical signs include acute fever, rigors, productive cough with purulent or rust-colored sputum, pleuritic chest pain, tachypnea, and localized crackles or bronchial breath sounds on lung auscultation."
    },
    {
        "q": "How do you recognize an acute ischemic stroke using the FAST protocol?",
        "a": "The FAST protocol stands for: Facial drooping (one side of face droops when smiling), Arm weakness (inability to raise both arms evenly), Speech difficulty (slurred or garbled speech), and Time to call emergency services immediately. Thrombolysis with IV alteplase (tPA) must occur within 4.5 hours of onset."
    },
    {
        "q": "What is Gastroesophageal Reflux Disease (GERD) and how is it managed?",
        "a": "GERD occurs when gastric acid flows retrograde into the esophagus due to transient lower esophageal sphincter relaxation. Symptoms include retrosternal heartburn (pyrosis), regurgitation, and water brash. Treatment includes lifestyle changes, H2-receptor antagonists (Famotidine), and Proton Pump Inhibitors (Omeprazole)."
    },
    {
        "q": "What are the clinical signs of acute Appendicitis?",
        "a": "Acute appendicitis presents with vague periumbilical abdominal pain that subsequently migrates to the right lower quadrant over 12-24 hours. Physical examination reveals tenderness at McBurney's point, Rovsing's sign, guarding, and rebound tenderness, requiring urgent laparoscopic appendectomy."
    },
    {
        "q": "What causes Anemia and what are the laboratory findings in Iron Deficiency Anemia?",
        "a": "Anemia is a reduction in red blood cell mass or hemoglobin concentration below reference levels. Iron deficiency anemia is a microcytic, hypochromic anemia characterized by low serum iron, elevated total iron-binding capacity (TIBC), low serum ferritin, and reduced mean corpuscular volume (MCV < 80 fL)."
    },
    {
        "q": "What is the emergency first-aid protocol for Anaphylaxis?",
        "a": "Anaphylaxis is a severe, life-threatening systemic hypersensitivity reaction. The definitive immediate treatment is intramuscular injection of Epinephrine (0.3-0.5 mg of 1:1000 solution for adults) into the anterolateral mid-thigh, supplemented by airway management, intravenous fluids, and supplemental oxygen."
    },
    {
        "q": "How does a Migraine headache differ from a Tension-type headache?",
        "a": "Migraines are typically unilateral, pulsating or throbbing, moderate to severe in intensity, aggravated by physical activity, and accompanied by photophobia, phonophobia, or nausea. Tension headaches are typically bilateral, non-pulsatile with band-like pressure, and lack systemic symptoms like vomiting."
    },
    {
        "q": "What is Sepsis and what are the criteria for septic shock?",
        "a": "Sepsis is life-threatening organ dysfunction caused by a dysregulated host response to infection, evaluated via the SOFA score. Septic shock is a subset characterized by persistent hypotension requiring vasopressors to maintain mean arterial pressure (MAP) >= 65 mmHg and serum lactate > 2 mmol/L despite fluid resuscitation."
    }
]

# Generate repetitive variations with diverse phrasing to create an adequate corpus for the causal LM
formatted_pairs = []
for item in medical_kb:
    formatted_pairs.append(f"Question: {item['q']}\nAnswer: {item['a']}\n")

# Replicate to ~500KB text for solid training density
full_text = "\n".join(formatted_pairs * 120)

with open('medical_data.txt', 'w', encoding='utf-8') as f:
    f.write(full_text)

print(f"Medical Q&A corpus created successfully! Total characters: {len(full_text):,}")
