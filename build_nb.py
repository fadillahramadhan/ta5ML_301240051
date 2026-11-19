import nbformat as nbf
nb = nbf.v4.new_notebook()
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s))
code = lambda s: c.append(nbf.v4.new_code_cell(s))

md("""# TA-05: Klasifikasi Kelangsungan Hidup Penumpang Titanic dengan Decision Tree
**Mata Praktikum:** Pembelajaran Mesin · **Modul 05:** Algoritma Decision Tree

Notebook ini berisi alur lengkap: EDA → preprocessing → Model A (tanpa batas / overfit) → Model B (pruning) → evaluasi → visualisasi pohon → feature importance → simpan model (Joblib).""")

code("""import os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sns.set_theme(style='whitegrid')
os.makedirs('../reports', exist_ok=True); os.makedirs('../model', exist_ok=True)""")

md("## 1. Memuat Dataset")
code("""df_raw = sns.load_dataset('titanic')
df_raw.to_csv('../data/titanic.csv', index=False)   # salinan lokal agar reproducible
print(df_raw.shape)
df_raw.head()""")

md("## 2. Eksplorasi Data (EDA)")
code("""fitur = ['pclass','sex','age','sibsp','parch','fare','embarked']
df = df_raw[['survived'] + fitur].copy()
print(df.dtypes, '\\n')
print('Missing values:'); print(df.isna().sum())
df.describe()""")
code("""fig, ax = plt.subplots(figsize=(5,3.5))
sns.countplot(x='survived', data=df, ax=ax)
ax.set_title('Distribusi Variabel Target'); ax.set_xticklabels(['Tidak Selamat (0)','Selamat (1)'])
plt.tight_layout(); plt.savefig('../reports/01_target.png', dpi=150); plt.show()
print(df['survived'].value_counts(normalize=True).round(3))""")
code("""fig, axes = plt.subplots(2, 3, figsize=(15, 8))
sns.barplot(x='sex', y='survived', data=df, ax=axes[0,0]);      axes[0,0].set_title('Survival rate vs Sex')
sns.barplot(x='pclass', y='survived', data=df, ax=axes[0,1]);   axes[0,1].set_title('Survival rate vs Pclass')
sns.barplot(x='embarked', y='survived', data=df, ax=axes[0,2]); axes[0,2].set_title('Survival rate vs Embarked')
sns.histplot(data=df, x='age', hue='survived', bins=30, multiple='stack', ax=axes[1,0]); axes[1,0].set_title('Distribusi Age')
sns.boxplot(x='survived', y='fare', data=df, ax=axes[1,1]); axes[1,1].set_yscale('log'); axes[1,1].set_title('Fare vs Survived (log)')
sns.barplot(x='sibsp', y='survived', data=df, ax=axes[1,2]);    axes[1,2].set_title('Survival rate vs SibSp')
plt.tight_layout(); plt.savefig('../reports/02_eda.png', dpi=150); plt.show()""")
md("**Temuan EDA:** perempuan jauh lebih banyak selamat daripada laki-laki, penumpang kelas 1 punya survival rate tertinggi, dan anak kecil cenderung lebih banyak selamat. Kelas target tidak seimbang ringan (± 62% tidak selamat vs 38% selamat).")

md("""## 3. Pra-pemrosesan Data
- **Missing value:** `age` (177 baris) → median; `embarked` (2 baris) → modus.
- **Fitur kategorikal:** `sex` dan `embarked` → One-Hot Encoding (`pd.get_dummies`).""")
code("""median_age = float(df['age'].median())
mode_embarked = df['embarked'].mode()[0]
df['age'] = df['age'].fillna(median_age)
df['embarked'] = df['embarked'].fillna(mode_embarked)
print('Sisa missing value:', int(df.isna().sum().sum()))

df_enc = pd.get_dummies(df, columns=['sex','embarked'], dtype=int)
X = df_enc.drop('survived', axis=1)
y = df_enc['survived']
print(list(X.columns))
X.head()""")

md("## 4. Split Data & Pelatihan Model")
code("""X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
print(X_train.shape, X_test.shape)

# Model A: tanpa batasan (potensi overfit)
model_a = DecisionTreeClassifier(max_depth=None, random_state=42).fit(X_train, y_train)
# Model B: pruning (pre-pruning dengan max_depth=3)
model_b = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
print('Kedalaman Model A:', model_a.get_depth(), '| Jumlah daun:', model_a.get_n_leaves())
print('Kedalaman Model B:', model_b.get_depth(), '| Jumlah daun:', model_b.get_n_leaves())""")

md("## 5. Evaluasi & Perbandingan Model A vs B")
code("""hasil = {}
for nama, m in [('Model A (max_depth=None)', model_a), ('Model B (max_depth=3)', model_b)]:
    pred = m.predict(X_test)
    hasil[nama] = dict(train=accuracy_score(y_train, m.predict(X_train)), test=accuracy_score(y_test, pred))
    print('='*60); print(nama)
    print('Confusion Matrix:'); print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=['Tidak Selamat','Selamat']))

cmp = pd.DataFrame(hasil).T
cmp['gap (train-test)'] = cmp['train'] - cmp['test']
cmp.round(4)""")
code("""fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (nama, m) in zip(axes, [('Model A (overfit)', model_a), ('Model B (pruned)', model_b)]):
    cm = confusion_matrix(y_test, m.predict(X_test))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Tidak Selamat','Selamat'], yticklabels=['Tidak Selamat','Selamat'])
    ax.set_title(nama); ax.set_xlabel('Prediksi'); ax.set_ylabel('Aktual')
plt.tight_layout(); plt.savefig('../reports/03_confusion.png', dpi=150); plt.show()""")
md("### Analisis Overfitting: kurva akurasi vs `max_depth`")
code("""depths = list(range(1, 19))
tr, te = [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, y_train)
    tr.append(m.score(X_train, y_train)); te.append(m.score(X_test, y_test))
plt.figure(figsize=(8,4.5))
plt.plot(depths, tr, 'o-', label='Akurasi Train'); plt.plot(depths, te, 's-', label='Akurasi Test')
plt.axvline(3, color='gray', ls='--', label='max_depth=3 (Model B)')
plt.xlabel('max_depth'); plt.ylabel('Akurasi'); plt.title('Overfitting: Train vs Test'); plt.legend()
plt.tight_layout(); plt.savefig('../reports/04_overfitting_curve.png', dpi=150); plt.show()""")
md("""**Catatan trade-off:** Model B sangat baik mengenali yang *tidak selamat* (recall 0,95) tetapi recall kelas *Selamat* hanya 0,54, sehingga model cenderung konservatif. Model A lebih seimbang di kelas Selamat (recall 0,64) namun akurasi keseluruhan lebih rendah dan tidak stabil.

**Analisis:** Model A tumbuh hingga kedalaman ±18 dan menghafal data latih (akurasi train ≈ 98%), tetapi akurasi test hanya ≈ 76%. Model B (depth=3) memiliki akurasi train ≈ 83% dan test ≈ 79,5%, sehingga selisih (gap) train–test (±3,5 poin vs ±21,8 poin pada Model A) jauh lebih kecil, artinya generalisasi lebih baik dan modelnya jauh lebih sederhana. Kurva menunjukkan akurasi test stagnan di kisaran 79% sejak depth 3–4 sementara akurasi train terus naik: tanda klasik overfitting.""")

md("## 6. Visualisasi Pohon (Model B)")
code("""plt.figure(figsize=(22, 10))
plot_tree(model_b, feature_names=X.columns, class_names=['Tidak Selamat','Selamat'],
          filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree Model B (max_depth=3)')
plt.savefig('../reports/05_tree_model_b.png', dpi=150, bbox_inches='tight'); plt.show()
print(export_text(model_b, feature_names=list(X.columns)))""")
md("""## 7. Interpretasi Aturan (jalur akar → daun)
1. **JIKA** penumpang bukan perempuan (`sex_female ≤ 0.5`) **DAN** usia > 3,5 tahun **MAKA** → **Tidak Selamat** (jalur terbesar; laki-laki dewasa).
2. **JIKA** penumpang bukan perempuan **DAN** usia ≤ 3,5 tahun **DAN** `sibsp` ≤ 3 **MAKA** → **Selamat** (anak laki-laki kecil diprioritaskan).
3. **JIKA** perempuan **DAN** kelas tiket 1 atau 2 (`pclass ≤ 2.5`) **MAKA** → **Selamat**.
4. **JIKA** perempuan **DAN** kelas 3 **DAN** `fare` > 7,89 **MAKA** → **Tidak Selamat** (perempuan kelas 3 dengan tarif lebih mahal, cenderung tinggal di dek bawah).

Aturan-aturan ini konsisten dengan fakta historis "perempuan dan anak-anak didahulukan" serta ketimpangan antar kelas tiket.""")

md("## 8. Feature Importance (Model B)")
code("""fi = pd.DataFrame({'Fitur': X.columns, 'Pentingnya': model_b.feature_importances_}).sort_values('Pentingnya', ascending=False)
plt.figure(figsize=(8,4.5))
sns.barplot(x='Pentingnya', y='Fitur', data=fi, hue='Fitur', palette='viridis', legend=False)
plt.title('Feature Importance (Model B)'); plt.tight_layout()
plt.savefig('../reports/06_feature_importance.png', dpi=150); plt.show()
fi.round(4)""")
md("Jenis kelamin (`sex_female`) mendominasi, diikuti `pclass`, `age`, dan `fare`. Fitur `parch`, `embarked_*` tidak dipakai oleh pohon dengan kedalaman 3.")

md("## 9. Simpan Model (Joblib)")
code("""joblib.dump(model_b, '../model/model_b.joblib')
meta = dict(features=list(X.columns), median_age=median_age, mode_embarked=mode_embarked,
            metrics=dict(train_acc=float(hasil['Model B (max_depth=3)']['train']),
                         test_acc=float(hasil['Model B (max_depth=3)']['test'])),
            max_depth=3)
json.dump(meta, open('../model/meta.json', 'w'), indent=2)
# uji muat ulang
m = joblib.load('../model/model_b.joblib')
print('Akurasi test setelah load:', round(m.score(X_test, y_test), 4))""")

nb["cells"] = c
nbf.write(nb, "notebook/TA05_Decision_Tree.ipynb")
