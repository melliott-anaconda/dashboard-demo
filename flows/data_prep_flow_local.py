"""
Fraud Detection - Data Preparation Flow (Local version - no @pypi)
"""
from metaflow import FlowSpec, step, Parameter, card, current

class FraudDataPrepFlow(FlowSpec):
    demo_mode = Parameter("demo_mode", help="Use reduced dataset", default=True, type=bool)

    @step
    def start(self):
        import pandas as pd
        import numpy as np
        import os

        possible_paths = [
            "data/creditcard.csv",
            os.path.expanduser("~/fraud-detection-ob/data/creditcard.csv"),
        ]
        self.data = None
        for path in possible_paths:
            if os.path.exists(path):
                self.data = pd.read_csv(path)
                print(f"Loaded from {path}: {len(self.data):,} transactions")
                break

        if self.data is None:
            print("CSV not found - generating synthetic dataset")
            np.random.seed(42)
            n_legit, n_fraud = 50000, 87
            all_features = np.vstack([np.random.randn(n_legit, 28)*0.5, np.random.randn(n_fraud, 28)*3.0])
            labels = np.array([0]*n_legit + [1]*n_fraud)
            times = np.random.randint(0, 172800, n_legit + n_fraud)
            amounts = np.concatenate([np.random.exponential(50, n_legit)+5, np.random.exponential(500, n_fraud)+100])
            cols = [f'V{i}' for i in range(1, 29)]
            self.data = pd.DataFrame(all_features, columns=cols)
            self.data['Time'] = times
            self.data['Amount'] = amounts
            self.data['Class'] = labels
            self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
            print(f"Synthetic dataset: {len(self.data):,} transactions")

        print(f"Fraud rate: {(self.data['Class']==1).mean()*100:.4f}%")
        self.next(self.add_features)

    @step
    def add_features(self):
        import numpy as np
        LEGIT = ["AMAZON.COM MKTP US","WALMART SUPERCENTER","STARBUCKS","SHELL GAS STATION","TARGET STORE","COSTCO WHOLESALE","NETFLIX SUBSCRIPTION","SPOTIFY PREMIUM","UBER TRIP","APPLE.COM BILL","WHOLE FOODS MARKET","CVS PHARMACY"]
        SUSP = ["BITCOIN ATM UNKNOWN","WIRE TRANSFER 7823","ONLINE CASINO DEPOSIT","CRYPTO EXCHANGE UNVERIFIED","UNKNOWN MERCHANT 9991","DARK WEB MARKET","HIGH RISK CASINO"]
        np.random.seed(42)
        self.data['merchant_description'] = [np.random.choice(SUSP) if c==1 else np.random.choice(LEGIT) for c in self.data['Class']]
        print(f"Merchant descriptions added ({len(self.data):,} rows)")
        self.next(self.split)

    @step
    def split(self):
        import numpy as np
        from sklearn.model_selection import train_test_split
        X = self.data.drop(['Class','merchant_description'], axis=1)
        y = self.data['Class']
        descs = self.data['merchant_description']
        amts = self.data['Amount']
        X_tr,X_te,y_tr,y_te,d_tr,d_te,a_tr,a_te = train_test_split(X,y,descs,amts,test_size=0.2,random_state=42,stratify=y)
        if self.demo_mode:
            n_tr,n_te = min(50000,len(X_tr)),min(10000,len(X_te))
            idx_tr = np.random.choice(len(X_tr),n_tr,replace=False)
            idx_te = np.random.choice(len(X_te),n_te,replace=False)
            X_tr,y_tr,d_tr,a_tr = X_tr.iloc[idx_tr],y_tr.iloc[idx_tr],d_tr.iloc[idx_tr],a_tr.iloc[idx_tr]
            X_te,y_te,d_te,a_te = X_te.iloc[idx_te],y_te.iloc[idx_te],d_te.iloc[idx_te],a_te.iloc[idx_te]
        self.X_train,self.X_test = X_tr,X_te
        self.y_train,self.y_test = y_tr,y_te
        self.desc_train,self.desc_test = d_tr,d_te
        self.amt_train,self.amt_test = a_tr,a_te
        print(f"Train: {len(self.X_train):,} ({(self.y_train==1).sum()} fraud)")
        print(f"Test:  {len(self.X_test):,} ({(self.y_test==1).sum()} fraud)")
        self.next(self.end)

    @card
    @step
    def end(self):
        print(f"Data Prep Complete: Train {len(self.X_train):,}, Test {len(self.X_test):,}")
        print(f"Next: python flows/training_flow_local.py run --data_run_id {current.run_id}")

if __name__ == "__main__":
    FraudDataPrepFlow()
