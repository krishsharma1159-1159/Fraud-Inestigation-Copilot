import os
import json
import joblib
import pandas as pd
import numpy as np
import shap

MODEL_PATH = os.environ.get("MODEL_PATH", r"e:\Skillcred\models\tuned\lightgbm_tuned.pkl")

class SHAPExplainer:
    """
    Dedicated SHAP Explainability layer using TreeExplainer for LightGBM fraud model.
    """
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.explainer = None
        self.expected_features = []
        self.base_value = None
        self.output_space = "raw margin (log-odds)"
        self.load_model()

    def load_model(self, model_path=None):
        if model_path:
            self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")
        
        self.model = joblib.load(self.model_path)
        self.expected_features = list(self.model.feature_name_)
        if len(self.expected_features) != 506:
            raise ValueError(f"Expected 506 features on model, found {len(self.expected_features)}")
        
        self.explainer = shap.TreeExplainer(self.model)
        ev = self.explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            self.base_value = float(ev[0])
        else:
            self.base_value = float(ev)

    def _validate_input_features(self, features_df: pd.DataFrame):
        if not isinstance(features_df, pd.DataFrame):
            raise TypeError(f"Input features must be a pandas DataFrame, got {type(features_df)}")
        if len(features_df.columns) != 506:
            raise ValueError(f"Input DataFrame must contain exactly 506 columns, got {len(features_df.columns)}")
        if list(features_df.columns) != self.expected_features:
            missing = [f for f in self.expected_features if f not in features_df.columns]
            extra = [f for f in features_df.columns if f not in self.expected_features]
            err = "Input DataFrame columns do not match expected 506 model features."
            if missing:
                err += f" Missing: {missing[:5]}"
            if extra:
                err += f" Extra: {extra[:5]}"
            raise ValueError(err)

    def explain_transaction(self, features_df: pd.DataFrame) -> dict:
        self._validate_input_features(features_df)
        
        exp = self.explainer(features_df)
        values = exp.values[0] # 1D numpy array of 506 float SHAP values
        
        feature_shaps = []
        row_vals = features_df.iloc[0].to_dict()

        for idx, feat_name in enumerate(self.expected_features):
            val = row_vals[feat_name]
            if pd.isna(val):
                py_val = None
            elif isinstance(val, (np.integer, int)):
                py_val = int(val)
            elif isinstance(val, (np.floating, float)):
                py_val = round(float(val), 6)
            else:
                py_val = str(val)

            s_val = float(values[idx])
            direction = "increases_model_output" if s_val > 0 else "decreases_model_output"

            feature_shaps.append({
                "feature": feat_name,
                "feature_value": py_val,
                "shap_value": round(s_val, 6),
                "absolute_shap_value": round(abs(s_val), 6),
                "direction": direction
            })

        return {
            "explainer": "SHAP TreeExplainer",
            "output_space": self.output_space,
            "base_value": round(self.base_value, 6),
            "all_features": feature_shaps
        }

    def get_top_features(self, features_df: pd.DataFrame, top_k: int = 10) -> dict:
        full_exp = self.explain_transaction(features_df)
        all_feats = full_exp["all_features"]
        
        sorted_feats = sorted(all_feats, key=lambda x: x["absolute_shap_value"], reverse=True)
        top_selected = sorted_feats[:top_k]

        shap_output = {
            "explainer": "SHAP TreeExplainer",
            "output_space": self.output_space,
            "base_value": full_exp["base_value"],
            "top_features": top_selected
        }

        self.validate_shap_output(shap_output)
        return shap_output

    def validate_shap_output(self, shap_output: dict) -> bool:
        if not isinstance(shap_output, dict):
            raise ValueError("SHAP output must be a dict.")
        if shap_output.get("explainer") != "SHAP TreeExplainer":
            raise ValueError(f"Invalid explainer name: {shap_output.get('explainer')}")
        if shap_output.get("output_space") != self.output_space:
            raise ValueError(f"Invalid output_space: {shap_output.get('output_space')}")
        if not isinstance(shap_output.get("base_value"), (float, int)):
            raise ValueError(f"Invalid base_value type: {type(shap_output.get('base_value'))}")

        top_feats = shap_output.get("top_features")
        if not isinstance(top_feats, list):
            raise ValueError("top_features must be a list.")

        prev_abs = float('inf')
        for item in top_feats:
            f_name = item.get("feature")
            if f_name not in self.expected_features:
                raise ValueError(f"Feature {f_name} not in 506 model feature list.")
            
            s_val = item.get("shap_value")
            a_val = item.get("absolute_shap_value")
            direction = item.get("direction")

            if not isinstance(s_val, (float, int)):
                raise ValueError(f"shap_value for {f_name} must be numeric, got {type(s_val)}")
            if not isinstance(a_val, (float, int)):
                raise ValueError(f"absolute_shap_value for {f_name} must be numeric, got {type(a_val)}")
            
            if abs(a_val - abs(s_val)) > 1e-5:
                raise ValueError(f"absolute_shap_value ({a_val}) != abs(shap_value) ({abs(s_val)}) for {f_name}")

            expected_dir = "increases_model_output" if s_val > 0 else "decreases_model_output"
            if direction != expected_dir:
                raise ValueError(f"direction '{direction}' does not match sign of shap_value {s_val} for {f_name}")

            if a_val > prev_abs + 1e-5:
                raise ValueError(f"top_features list is not properly sorted by absolute SHAP value: {a_val} > {prev_abs}")
            prev_abs = a_val

            desc_str = json.dumps(item).lower()
            for forbidden in ["causes fraud", "prevents fraud", "proves fraud", "fraudulent", "criminal"]:
                if forbidden in desc_str:
                    raise ValueError(f"SHAP explanation contains forbidden subjective claim: '{forbidden}'")

        return True

    def get_global_feature_importance(self, sample_df: pd.DataFrame, top_k: int = 20) -> list:
        self._validate_input_features(sample_df)
        shap_vals = self.explainer.shap_values(sample_df)
        
        if isinstance(shap_vals, list):
            vals = shap_vals[1]
        else:
            vals = shap_vals

        mean_abs = np.mean(np.abs(vals), axis=0)
        
        importance_list = []
        for idx, f_name in enumerate(self.expected_features):
            importance_list.append({
                "feature": f_name,
                "mean_absolute_shap": round(float(mean_abs[idx]), 6)
            })

        importance_list.sort(key=lambda x: x["mean_absolute_shap"], reverse=True)
        return importance_list[:top_k]

_shap_explainer_instance = None

def get_shap_explainer():
    global _shap_explainer_instance
    if _shap_explainer_instance is None:
        _shap_explainer_instance = SHAPExplainer()
    return _shap_explainer_instance
