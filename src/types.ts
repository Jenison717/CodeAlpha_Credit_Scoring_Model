export interface CreditDataRecord {
  id: number;
  income: number;          // in thousands, e.g., 20 to 150
  age: number;             // 18 to 75
  debt: number;            // debt-to-income ratio in % (0 to 100)
  payment_history: number; // score from 0 to 100 (delayed payments index)
  loan_amount: number;     // in thousands, e.g., 5 to 100
  employment_status: 'Unemployed' | 'Employed' | 'Self-Employed';
  credit_risk: 0 | 1;      // 0 = Bad (High Risk), 1 = Good (Low Risk)
}

export interface ModelHyperparameters {
  logisticRegression: {
    learningRate: number;
    iterations: number;
    regularization: 'none' | 'l2';
    lambda: number;
  };
  decisionTree: {
    maxDepth: number;
    minSamplesSplit: number;
  };
  randomForest: {
    numTrees: number;
    maxDepth: number;
    featureSubsampleRatio: number;
  };
}

export interface EvaluationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  auc: number;
}

export interface ROCPoint {
  fpr: number;
  tpr: number;
}

export interface ModelResults {
  name: string;
  metrics: EvaluationMetrics;
  rocCurve: ROCPoint[];
  featureImportance: { feature: string; value: number }[];
}

export interface DatasetScenario {
  id: string;
  name: string;
  description: string;
  avgIncome: number;       // average income multiplier
  defaultRate: number;     // baseline default rate
  noiseLevel: number;      // random noise factor
}
