"""
Fraud Detection - Training Flow (Local version - no @pypi)
"""
from metaflow import FlowSpec, step, Parameter, card, resources, current

class FraudTrainingFlow(FlowSpec):
    data_run_id = Parameter("data_run_id", help="Run ID from FraudDataPrepFlow", required=True)

    @step
    def start(self):
        from metaflow import Flow
        run = Flow("FraudDataPrepFlow")[self.data_run_id]
        end = run["end"].task
        self.X_train = end.data.X_train
        self.X_test = end.data.X_test
        self.y_train = end.data.y_train
        self.y_test = end.data.y_test
        self.desc_train = end.data.desc_train
        self.desc_test = end.data.desc_test
        self.amt_train = end.data.amt_train
        self.amt_test = end.data.amt_test
        print(f"Loaded: Train {len(self.X_train):,}, Test {len(self.X_test):,}")
        self.hp_grid = [
            {"n_estimators": 100, "max_depth": 12},
            {"n_estimators": 200, "max_depth": 10},
            {"n_estimators": 200, "max_depth": 15},
            {"n_estimators": 300, "max_depth": 12},
        ]
        self.next(self.train, foreach="hp_grid")

    @step
    def train(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
        self.hparams = self.input
        print(f"Training: {self.hparams}")
        self.rf_model = RandomForestClassifier(n_estimators=self.hparams["n_estimators"], max_depth=self.hparams["max_depth"], random_state=42, n_jobs=-1)
        self.rf_model.fit(self.X_train, self.y_train)
        y_prob = self.rf_model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(self.y_test, y_pred, average="binary")
        self.metrics = {"precision": round(prec,4), "recall": round(rec,4), "f1": round(f1,4), "auc_roc": round(roc_auc_score(self.y_test, y_prob),4), "avg_precision": round(average_precision_score(self.y_test, y_prob),4)}
        print(f"  F1={self.metrics['f1']}, AUC={self.metrics['auc_roc']}")
        self.next(self.select_best)

    @step
    def select_best(self, inputs):
        self.merge_artifacts(inputs, include=["X_train","X_test","y_train","y_test","desc_train","desc_test","amt_train","amt_test"])
        best = max(inputs, key=lambda x: x.metrics["f1"])
        self.best_model = best.rf_model
        self.best_metrics = best.metrics
        self.best_hparams = best.hparams
        self.model_config = {"llm_threshold":0.3,"xgb_weight":0.6,"llm_weight":0.4,"block_threshold":0.8,"review_threshold":0.5}
        print(f"\n{'='*70}\n  RESULTS\n{'='*70}")
        for inp in inputs:
            m = " <-- BEST" if inp.hparams == best.hparams else ""
            print(f"  {inp.hparams} -> F1={inp.metrics['f1']}, AUC={inp.metrics['auc_roc']}{m}")
        self.next(self.end)

    @card
    @step
    def end(self):
        print(f"Done: F1={self.best_metrics['f1']}, AUC={self.best_metrics['auc_roc']}")
        print(f"Next: python flows/scoring_flow_local.py run --training_run_id {current.run_id}")

if __name__ == "__main__":
    FraudTrainingFlow()
