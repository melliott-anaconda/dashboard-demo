"""
Fraud Detection - Scoring Flow (Kubernetes-compatible)
"""
from metaflow import FlowSpec, step, Parameter, card, resources, current, pypi_base

DEPS = {"pandas": "2.3.0", "numpy": "2.2.6", "scikit-learn": "1.7.2"}

@pypi_base(python='3.12', packages=DEPS)
class FraudScoringFlow(FlowSpec):
    training_run_id = Parameter("training_run_id", help="Run ID from FraudTrainingFlow", required=True)

    @step
    def start(self):
        from metaflow import Flow
        run = Flow("FraudTrainingFlow")[self.training_run_id]
        end = run["end"].task
        self.model = end.data.best_model
        self.best_metrics = end.data.best_metrics
        self.model_config = end.data.model_config
        print(f"Loaded model from run {self.training_run_id}: {type(self.model).__name__}")
        print(f"Metrics: {self.best_metrics}")
        self.next(self.load_transactions)

    @step
    def load_transactions(self):
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        legit_m = ["AMAZON.COM MKTP US","WALMART SUPERCENTER","STARBUCKS","SHELL GAS STATION","TARGET STORE","COSTCO WHOLESALE","NETFLIX SUBSCRIPTION","SPOTIFY PREMIUM","UBER TRIP","APPLE.COM BILL","WHOLE FOODS MARKET","CVS PHARMACY"]
        susp_m = ["BITCOIN ATM UNKNOWN","WIRE TRANSFER 7823","ONLINE CASINO DEPOSIT","CRYPTO EXCHANGE UNVERIFIED","UNKNOWN MERCHANT 9991"]
        merchants, amounts, is_fraud, features_list = [], [], [], []
        feature_cols = [f'V{i}' for i in range(1,29)] + ['Time','Amount']
        for _ in range(90):
            merchants.append(np.random.choice(legit_m))
            amt = round(np.random.exponential(50)+5, 2)
            amounts.append(amt)
            is_fraud.append(0)
            features_list.append(np.append(np.random.randn(28)*0.5, [np.random.randint(0,172800), amt]))
        for _ in range(10):
            merchants.append(np.random.choice(susp_m))
            amt = round(np.random.exponential(500)+100, 2)
            amounts.append(amt)
            is_fraud.append(1)
            features_list.append(np.append(np.random.randn(28)*3.0, [np.random.randint(0,172800), amt]))
        self.X_batch = pd.DataFrame(features_list, columns=feature_cols)
        self.merchants = merchants
        self.amounts = amounts
        self.true_labels = is_fraud
        print(f"Batch: 100 transactions (90 legit, 10 suspicious)")
        self.next(self.score)

    @resources(cpu=2, memory=4096)
    @step
    def score(self):
        import numpy as np
        import pandas as pd
        block_t = self.model_config["block_threshold"]
        review_t = self.model_config["review_threshold"]
        fraud_scores = self.model.predict_proba(self.X_batch)[:, 1]
        decisions = []
        for s in fraud_scores:
            if s > block_t: decisions.append("BLOCK")
            elif s > review_t: decisions.append("REVIEW")
            else: decisions.append("APPROVE")
        self.results_df = pd.DataFrame({'Merchant':self.merchants,'Amount':[f'${a:.2f}' for a in self.amounts],'Score':[f'{s:.4f}' for s in fraud_scores],'Decision':decisions,'True_Label':self.true_labels})
        self.fraud_scores = fraud_scores.tolist()
        self.decisions = decisions
        self.next(self.summarize)

    @card
    @step
    def summarize(self):
        import pandas as pd
        import numpy as np
        counts = pd.Series(self.decisions).value_counts()
        predicted = np.array([1 if d!='APPROVE' else 0 for d in self.decisions])
        actual = np.array(self.true_labels)
        print(f"\n{'='*70}\n  BATCH SUMMARY\n{'='*70}")
        print(f"  Total: {len(self.results_df)}")
        print(f"  Approved: {counts.get('APPROVE',0)}")
        print(f"  Review:   {counts.get('REVIEW',0)}")
        print(f"  Blocked:  {counts.get('BLOCK',0)}")
        print(f"  Accuracy: {(predicted==actual).mean():.4f}")
        print(f"  True frauds: {actual.sum()}, Flagged: {predicted.sum()}")
        flagged = self.results_df[self.results_df['Decision'].isin(['BLOCK','REVIEW'])]
        if len(flagged) > 0:
            print(f"\nFlagged:\n{flagged.to_string(index=False)}")
        self.next(self.end)

    @step
    def end(self):
        print(f"Scoring complete. View at: https://ui.merced.obp.outerbounds.com/p/default/FraudScoringFlow/{current.run_id}")

if __name__ == "__main__":
    FraudScoringFlow()
