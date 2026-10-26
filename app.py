from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="SafeRoute | Accident Severity", page_icon="🚦", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.4rem;max-width:1320px}
.hero{background:linear-gradient(120deg,#102a43,#176b73);color:white;padding:1.7rem 2rem;border-radius:18px;margin-bottom:1rem}
.hero h1{color:white;margin:0;font-size:2.15rem}.hero p{color:#e1edf1;margin:.4rem 0 0}
div[data-testid="stMetric"]{background:#f3f7fa;border:1px solid #e0e8ef;padding:13px 15px;border-radius:14px}
</style>""",unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🚦 SafeRoute</h1><p>Road accident severity prediction · Python Random Forest</p></div>',unsafe_allow_html=True)

ROOT=Path(__file__).resolve().parent
DATA_PATH=ROOT/"data"/"accident_ppt_summary.csv"
MODEL_PATH=ROOT/"models"/"accident_severity_random_forest.joblib"
labels_map={"accidents_fatal":"Fatal","accidents_grievous_injury":"Grievous Injury","accidents_minor_injury":"Minor Injury","accidents_no_injury":"No Injury"}
dimensions={1:"Overall",4:"Month",5:"Weather season",6:"Time of day",7:"Collision type",8:"Accident nature",10:"Traffic violation",13:"Road direction",14:"Impacting vehicle",15:"Impacted vehicle",16:"Junction type"}

test_fraction=0.25

uploaded_df=None
mode="PowerPoint case counts"
summary_raw=None
if uploaded_df is None:
    if not DATA_PATH.exists():
        st.error(f"PowerPoint summary not found: {DATA_PATH}")
        st.stop()
    summary_raw=pd.read_csv(DATA_PATH)
    summary_raw.columns=summary_raw.columns.str.strip().str.lower()
    summary_raw["slide"]=pd.to_numeric(summary_raw["slide"],errors="coerce")
    summary_raw["value"]=pd.to_numeric(summary_raw["value"],errors="coerce")
    summary_raw["year"]=summary_raw["year"].fillna("").astype(str).str.strip()
    summary_raw["category"]=summary_raw["category"].fillna("").astype(str).str.strip()
    summary_raw["metric"]=summary_raw["metric"].fillna("").astype(str).str.strip().str.lower()
    grouped=summary_raw[summary_raw["slide"].isin(dimensions)&summary_raw["metric"].isin(labels_map)&summary_raw["value"].notna()].copy()
    grouped["breakdown"]=grouped["slide"].map(dimensions)
    grouped["severity"]=grouped["metric"].map(labels_map)
    grouped["category"]=grouped["category"].replace("","Unspecified")
    grouped["year"]=grouped["year"].replace("","2023-2025")
    cases=grouped.groupby(["breakdown","category","year","severity"],as_index=False)["value"].sum()
    cases=cases[cases["value"]>0].copy()
    if cases.empty:
        st.error("No severity case counts were found in the PowerPoint CSV.")
        st.stop()
    model_features=["breakdown","category","year"]
    target="severity"
    counts=cases["value"].round().astype(int)
    expanded=cases.loc[cases.index.repeat(counts)].reset_index(drop=True)
    X=expanded[model_features].astype(str)
    y=expanded["severity"].astype(str)
    groups=cases["breakdown"].astype(str)+" | "+cases["category"].astype(str)
    try:
        split=GroupShuffleSplit(n_splits=1,test_size=test_fraction,random_state=42)
        tr,te=next(split.split(cases[model_features],cases["severity"],groups=groups))
        train_cells,test_cells=cases.iloc[tr],cases.iloc[te]
        train_rows=train_cells.loc[train_cells.index.repeat(train_cells["value"].round().astype(int))].reset_index(drop=True)
        test_rows=test_cells.loc[test_cells.index.repeat(test_cells["value"].round().astype(int))].reset_index(drop=True)
        X_train,y_train=train_rows[model_features].astype(str),train_rows["severity"].astype(str)
        X_test,y_test=test_rows[model_features].astype(str),test_rows["severity"].astype(str)
    except ValueError:
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_fraction,random_state=42,stratify=y)
    dataset_view=cases[["breakdown","category","year","severity","value"]].rename(columns={"value":"case_count"})
    dataset_view["severity"]=dataset_view["severity"].astype(str)
    display_rows=int(cases["value"].sum())
num_cols=X_train.select_dtypes(include=["number"]).columns.tolist()
cat_cols=[c for c in model_features if c not in num_cols]
transformers=[]
if num_cols:
    transformers.append(("numeric",Pipeline([("fill",SimpleImputer(strategy="median"))]),num_cols))
if cat_cols:
    transformers.append(("categorical",Pipeline([("fill",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore"))]),cat_cols))
preprocessor=ColumnTransformer(transformers=transformers)
model=Pipeline([("prepare",preprocessor),("forest",RandomForestClassifier(class_weight="balanced_subsample",random_state=42,min_samples_leaf=2,n_jobs=-1))])
try:
    model.fit(X_train,y_train)
    test_prediction=model.predict(X_test)
except Exception as exc:
    st.error(f"Could not train the model. Check the selected columns. Details: {exc}")
    st.stop()
score=accuracy_score(y_test,test_prediction)
try:
    eval_x=X_test.copy()
    eval_y=y_test.copy()
    if len(eval_x)>1000:
        sample=eval_x.sample(1000,random_state=42).index
        eval_x=eval_x.loc[sample]
        eval_y=eval_y.loc[sample]
    pi=permutation_importance(model,eval_x,eval_y,n_repeats=3,random_state=42,scoring="accuracy",n_jobs=-1)
    feature_importance=pd.Series(pi.importances_mean,index=model_features).sort_values(ascending=False)
except Exception:
    feature_importance=pd.Series(dtype=float)
# Refit on all available training data for interactive prediction.
model.fit(X,y)
MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)
joblib.dump({"model":model,"features":model_features,"target":target},MODEL_PATH)

st.caption(f"Training source: {mode} · {display_rows:,} cases/records · outcome: {target}")
overview,predict_tab,insights,data_tab=st.tabs(["Overview","Severity predictor","Model insights","Data"])
with overview:
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Cases / records",f"{display_rows:,}")
    m2.metric("Severity classes",f"{y.nunique():,}")
    m3.metric("Test accuracy",f"{score:.1%}")
    m4.metric("Test records",f"{len(y_test):,}")
    st.subheader("Severity distribution")
    if mode=="PowerPoint case counts":
        dist=cases.groupby("severity")["value"].sum().sort_values(ascending=False)
        st.bar_chart(dist.rename_axis("Severity").rename("Cases"))
        annual=summary_raw[(summary_raw["slide"]==1)&(summary_raw["metric"]=="accidents_total")&summary_raw["year"].isin(["2023","2024","2025"])]
        annual=annual.drop_duplicates("year").set_index("year")["value"]
        year_cols=st.columns(3)
        for col,yr in zip(year_cols,["2023","2024","2025"]):
            col.metric(f"Reported accidents · {yr}",f"{int(annual.get(yr,0)):,}")
    else:
        dist=df[target].value_counts()
        st.bar_chart(dist.rename_axis("Severity").rename("Records"))
with predict_tab:
    st.subheader("Predict severity for a new accident")
    st.write("Enter the information available before or during the event.")
    values={}
    if mode=="PowerPoint case counts":
        a,b=st.columns(2)
        breakdown=a.selectbox("Road factor",sorted(cases["breakdown"].unique()))
        sub=cases[cases["breakdown"]==breakdown]
        category=b.selectbox("Recorded category",sorted(sub["category"].unique()))
        years=sorted(sub[sub["category"]==category]["year"].unique())
        year=st.selectbox("Year / period",years)
        values={"breakdown":breakdown,"category":category,"year":year}
        submitted=st.button("Predict severity",type="primary")
    else:
        with st.form("new_accident"):
            cols=st.columns(2)
            for i,c in enumerate(model_features):
                s=X[c]
                label=c.replace("_"," ").title()
                with cols[i%2]:
                    if c in num_cols:
                        obs=pd.to_numeric(s,errors="coerce").dropna()
                        default=float(obs.median()) if len(obs) else 0.0
                        lo=float(obs.min()) if len(obs) else 0.0
                        hi=float(obs.max()) if len(obs) else max(default+1,1)
                        if hi<=lo: hi=lo+1
                        values[c]=st.number_input(label,min_value=lo,max_value=hi,value=min(max(default,lo),hi))
                    else:
                        opts=sorted(s.dropna().astype(str).unique().tolist())
                        if len(opts)<=30:
                            pick=st.selectbox(label,["Other…"]+opts,key=f"pick_{c}")
                            values[c]=st.text_input(f"Enter {label.lower()}",key=f"other_{c}") if pick=="Other…" else pick
                        else:
                            values[c]=st.text_input(label)
            submitted=st.form_submit_button("Predict severity",type="primary")
    if submitted:
        query=pd.DataFrame([{c:values[c] for c in model_features}],columns=model_features)
        prediction=model.predict(query)[0]
        proba=model.predict_proba(query)[0]
        classes=model.named_steps["forest"].classes_
        order=np.argsort(proba)[::-1]
        p1,p2=st.columns([1,1.2])
        p1.success(f"Predicted severity: **{prediction}**")
        p1.metric("Top class score",f"{proba[order[0]]:.1%}")
        p2.bar_chart(pd.Series({str(classes[i]):float(proba[i]) for i in order},name="Model score"))
with insights:
    st.subheader("Evaluation on held-out data")
    st.metric("Accuracy",f"{score:.1%}")
    class_names=sorted(set(y_test.astype(str))|set(pd.Series(test_prediction).astype(str)))
    cm=confusion_matrix(y_test.astype(str),pd.Series(test_prediction).astype(str),labels=class_names)
    st.write("Confusion matrix")
    st.dataframe(pd.DataFrame(cm,index=class_names,columns=class_names).rename_axis("Actual").rename_axis("Predicted",axis=1),use_container_width=True)
    st.write("Classification report")
    report=classification_report(y_test,test_prediction,output_dict=True,zero_division=0)
    st.dataframe(pd.DataFrame(report).T,use_container_width=True)
    st.write("Feature importance")
    if not feature_importance.empty:
        st.bar_chart(feature_importance.rename("Importance"))
    else:
        st.caption("Feature importance could not be calculated for this test split.")
with data_tab:
    st.subheader("Loaded data")
    st.dataframe(dataset_view,use_container_width=True,height=430)
    st.download_button("Download loaded data as CSV",dataset_view.to_csv(index=False).encode("utf-8"),"accident_records_used.csv","text/csv")






