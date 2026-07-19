import { CreditDataRecord, EvaluationMetrics, ModelHyperparameters, ModelResults, ROCPoint } from './types';

// ==========================================
// 1. DATA PREPROCESSING & UTILITIES
// ==========================================

export interface PreprocessedFeatures {
  id: number;
  // Features: [income, age, debt, payment_history, loan_amount, emp_employed, emp_self_employed, emp_unemployed]
  x: number[]; 
  y: number; // 0 or 1
}

/**
 * Preprocesses raw CreditDataRecords.
 * - Handles missing values (by mean/mode imputation)
 * - Encodes categorical columns (employment_status into 3 binary features)
 * - Scales numerical features using Z-score standardization
 */
export function preprocessData(
  rawRecords: CreditDataRecord[],
  trainRatio: number = 0.7,
  injectMissingForDemo: boolean = true
): {
  trainSet: PreprocessedFeatures[];
  testSet: PreprocessedFeatures[];
  featureNames: string[];
  preprocessingLog: string[];
} {
  const log: string[] = [];
  log.push(`Starting preprocessing of ${rawRecords.length} records.`);

  // Create a deep copy to prevent mutating raw records
  const records = rawRecords.map(r => ({ ...r }));

  // Demonstrate Imputation: Artificially inject missing values if enabled
  if (injectMissingForDemo && records.length > 10) {
    // Inject 3 missing incomes and 3 missing ages
    log.push("Injecting 6 artificial missing values for preprocessing demonstration:");
    const injectIndices = [5, 12, 28];
    injectIndices.forEach(idx => {
      if (idx < records.length) {
        (records[idx] as any).income = null;
        log.push(` - Set record #${records[idx].id} income = null`);
      }
    });
    const injectIndicesAge = [8, 35, 47];
    injectIndicesAge.forEach(idx => {
      if (idx < records.length) {
        (records[idx] as any).age = null;
        log.push(` - Set record #${records[idx].id} age = null`);
      }
    });
  }

  // Imputation calculations
  let sumIncome = 0;
  let countIncome = 0;
  let sumAge = 0;
  let countAge = 0;

  records.forEach(r => {
    if (r.income !== null && r.income !== undefined && !isNaN(r.income)) {
      sumIncome += r.income;
      countIncome++;
    }
    if (r.age !== null && r.age !== undefined && !isNaN(r.age)) {
      sumAge += r.age;
      countAge++;
    }
  });

  const meanIncome = countIncome > 0 ? sumIncome / countIncome : 50;
  const meanAge = countAge > 0 ? sumAge / countAge : 40;

  let imputedIncomeCount = 0;
  let imputedAgeCount = 0;

  records.forEach(r => {
    if (r.income === null || r.income === undefined || isNaN(r.income)) {
      r.income = Math.round(meanIncome);
      imputedIncomeCount++;
    }
    if (r.age === null || r.age === undefined || isNaN(r.age)) {
      r.age = Math.round(meanAge);
      imputedAgeCount++;
    }
  });

  if (imputedIncomeCount > 0 || imputedAgeCount > 0) {
    log.push(`Imputation: Imputed ${imputedIncomeCount} income values with mean (${meanIncome.toFixed(1)})`);
    log.push(`Imputation: Imputed ${imputedAgeCount} age values with mean (${meanAge.toFixed(1)})`);
  }

  // Get statistics for scaling numerical features:
  // numericals = [income, age, debt, payment_history, loan_amount]
  const numericalKeys: (keyof CreditDataRecord)[] = ['income', 'age', 'debt', 'payment_history', 'loan_amount'];
  const means: Record<string, number> = {};
  const stds: Record<string, number> = {};

  numericalKeys.forEach(key => {
    const vals = records.map(r => r[key] as number);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const variance = vals.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / vals.length;
    const std = Math.sqrt(variance) || 1; // avoid divide by zero
    means[key as string] = mean;
    stds[key as string] = std;
  });

  log.push("Standard Scaling Applied to Numerical Columns:");
  numericalKeys.forEach(k => {
    log.push(` - ${String(k)}: mean = ${means[k as string].toFixed(1)}, std = ${stds[k as string].toFixed(1)}`);
  });

  // Categorical encoding (One-Hot Encoding of employment_status)
  // Categories: 'Employed', 'Self-Employed', 'Unemployed'
  log.push("One-Hot Encoding Applied to 'employment_status':");
  log.push(" - Encoded as 3 columns: [is_employed, is_self_employed, is_unemployed]");

  const processed: PreprocessedFeatures[] = records.map(r => {
    // 1. Scale numericals
    const scaledIncome = ((r.income as number) - means.income) / stds.income;
    const scaledAge = ((r.age as number) - means.age) / stds.age;
    const scaledDebt = ((r.debt as number) - means.debt) / stds.debt;
    const scaledHistory = ((r.payment_history as number) - means.payment_history) / stds.payment_history;
    const scaledLoan = ((r.loan_amount as number) - means.loan_amount) / stds.loan_amount;

    // 2. Encode categorical
    const isEmployed = r.employment_status === 'Employed' ? 1 : 0;
    const isSelfEmployed = r.employment_status === 'Self-Employed' ? 1 : 0;
    const isUnemployed = r.employment_status === 'Unemployed' ? 1 : 0;

    return {
      id: r.id,
      x: [
        scaledIncome,
        scaledAge,
        scaledDebt,
        scaledHistory,
        scaledLoan,
        isEmployed,
        isSelfEmployed,
        isUnemployed
      ],
      y: r.credit_risk
    };
  });

  const featureNames = [
    'income',
    'age',
    'debt',
    'payment_history',
    'loan_amount',
    'emp_employed',
    'emp_self_employed',
    'emp_unemployed'
  ];

  // Train / Test split
  // For repeatability, we split deterministically based on record index
  // (e.g. taking every nth item or shuffling with a local generator, we will shuffle deterministically)
  const shuffled = [...processed];
  let localSeed = 12345;
  const rand = () => {
    const x = Math.sin(localSeed++) * 10000;
    return x - Math.floor(x);
  };
  
  // Fisher-Yates shuffle with seed
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const temp = shuffled[i];
    shuffled[i] = shuffled[j];
    shuffled[j] = temp;
  }

  const splitIdx = Math.floor(shuffled.length * trainRatio);
  const trainSet = shuffled.slice(0, splitIdx);
  const testSet = shuffled.slice(splitIdx);

  log.push(`Dataset split complete:`);
  log.push(` - Training set: ${trainSet.length} samples (${Math.round(trainRatio * 100)}%)`);
  log.push(` - Testing set: ${testSet.length} samples (${Math.round((1 - trainRatio) * 100)}%)`);

  return {
    trainSet,
    testSet,
    featureNames,
    preprocessingLog: log
  };
}


// ==========================================
// 2. MODEL 1: LOGISTIC REGRESSION (L2 REGULARIZED)
// ==========================================

export class LogisticRegression {
  private weights: number[] = [];
  private bias: number = 0;
  private learningRate: number;
  private iterations: number;
  private regularization: 'none' | 'l2';
  private lambda: number;

  constructor(lr = 0.1, iter = 100, reg: 'none' | 'l2' = 'l2', lambda = 0.1) {
    this.learningRate = lr;
    this.iterations = iter;
    this.regularization = reg;
    this.lambda = lambda;
  }

  private sigmoid(z: number): number {
    return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, z))));
  }

  public train(data: PreprocessedFeatures[], featureCount: number) {
    // Initialize weights and bias to 0
    this.weights = new Array(featureCount).fill(0);
    this.bias = 0;
    const m = data.length;

    for (let iter = 0; iter < this.iterations; iter++) {
      const dw = new Array(featureCount).fill(0);
      let db = 0;

      for (let i = 0; i < m; i++) {
        const xi = data[i].x;
        const yi = data[i].y;

        // Calculate hypothesis
        let z = this.bias;
        for (let j = 0; j < featureCount; j++) {
          z += xi[j] * this.weights[j];
        }
        const h = this.sigmoid(z);
        const error = h - yi;

        // Gradient accumulation
        for (let j = 0; j < featureCount; j++) {
          dw[j] += error * xi[j];
        }
        db += error;
      }

      // Update weights and bias
      for (let j = 0; j < featureCount; j++) {
        dw[j] /= m;
        if (this.regularization === 'l2') {
          // Add L2 penalty derivative
          dw[j] += (this.lambda / m) * this.weights[j];
        }
        this.weights[j] -= this.learningRate * dw[j];
      }
      this.bias -= this.learningRate * (db / m);
    }
  }

  public predictProb(x: number[]): number {
    let z = this.bias;
    for (let j = 0; j < x.length; j++) {
      z += x[j] * this.weights[j];
    }
    return this.sigmoid(z);
  }

  public getWeights(): number[] {
    return this.weights;
  }
}


// ==========================================
// 3. MODEL 2: DECISION TREE CLASSIFIER
// ==========================================

interface DecisionNode {
  featureIdx?: number;
  threshold?: number;
  left?: DecisionNode;
  right?: DecisionNode;
  value?: number; // predicted probability of class 1 if leaf
  isLeaf: boolean;
}

export class DecisionTree {
  private root: DecisionNode | null = null;
  private maxDepth: number;
  private minSamplesSplit: number;

  constructor(maxDepth = 5, minSamplesSplit = 4) {
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
  }

  private gini(data: PreprocessedFeatures[]): number {
    const total = data.length;
    if (total === 0) return 0;
    const count1 = data.filter(d => d.y === 1).length;
    const p1 = count1 / total;
    const p0 = 1 - p1;
    return 1 - (p0 * p0 + p1 * p1);
  }

  private split(data: PreprocessedFeatures[], featureIdx: number, threshold: number) {
    const left: PreprocessedFeatures[] = [];
    const right: PreprocessedFeatures[] = [];
    data.forEach(d => {
      if (d.x[featureIdx] <= threshold) {
        left.push(d);
      } else {
        right.push(d);
      }
    });
    return { left, right };
  }

  private buildTree(data: PreprocessedFeatures[], depth: number): DecisionNode {
    const numSamples = data.length;
    const numFeatures = data[0]?.x.length || 0;
    const count1 = data.filter(d => d.y === 1).length;
    const leafVal = numSamples > 0 ? count1 / numSamples : 0.5;

    // Base cases: pure node, depth limit, or small samples
    if (
      depth >= this.maxDepth ||
      numSamples < this.minSamplesSplit ||
      count1 === 0 ||
      count1 === numSamples
    ) {
      return { isLeaf: true, value: leafVal };
    }

    let bestGiniGain = -1;
    let bestFeatureIdx = -1;
    let bestThreshold = 0;
    let bestLeft: PreprocessedFeatures[] = [];
    let bestRight: PreprocessedFeatures[] = [];

    const currentGini = this.gini(data);

    // Grid search for best binary split
    for (let f = 0; f < numFeatures; f++) {
      // Collect unique threshold candidates from this feature
      const values = Array.from(new Set(data.map(d => d.x[f]))).sort((a, b) => a - b);
      // Try midpoints of adjacent sorted values
      for (let v = 0; v < values.length - 1; v++) {
        const threshold = (values[v] + values[v + 1]) / 2;
        const { left, right } = this.split(data, f, threshold);

        if (left.length === 0 || right.length === 0) continue;

        const leftGini = this.gini(left);
        const rightGini = this.gini(right);
        const childGiniWeighted = (left.length / numSamples) * leftGini + (right.length / numSamples) * rightGini;
        const giniGain = currentGini - childGiniWeighted;

        if (giniGain > bestGiniGain) {
          bestGiniGain = giniGain;
          bestFeatureIdx = f;
          bestThreshold = threshold;
          bestLeft = left;
          bestRight = right;
        }
      }
    }

    // If no gain was achieved, return leaf
    if (bestGiniGain <= 0 || bestLeft.length === 0 || bestRight.length === 0) {
      return { isLeaf: true, value: leafVal };
    }

    const leftChild = this.buildTree(bestLeft, depth + 1);
    const rightChild = this.buildTree(bestRight, depth + 1);

    return {
      isLeaf: false,
      featureIdx: bestFeatureIdx,
      threshold: bestThreshold,
      left: leftChild,
      right: rightChild
    };
  }

  public train(data: PreprocessedFeatures[]) {
    this.root = this.buildTree(data, 0);
  }

  private predictNode(node: DecisionNode, x: number[]): number {
    if (node.isLeaf) {
      return node.value!;
    }
    const val = x[node.featureIdx!];
    if (val <= node.threshold!) {
      return this.predictNode(node.left!, x);
    } else {
      return this.predictNode(node.right!, x);
    }
  }

  public predictProb(x: number[]): number {
    if (!this.root) return 0.5;
    return this.predictNode(this.root, x);
  }

  // Generate feature importances by counting node splits and Gini gains
  public getFeatureImportance(featureCount: number): number[] {
    const importances = new Array(featureCount).fill(0);
    const traverse = (node: DecisionNode) => {
      if (node.isLeaf || node.featureIdx === undefined) return;
      importances[node.featureIdx] += 1; // Simplified node split count
      traverse(node.left!);
      traverse(node.right!);
    };
    if (this.root) traverse(this.root);
    
    const sum = importances.reduce((a, b) => a + b, 0);
    return sum > 0 ? importances.map(v => v / sum) : importances;
  }
}


// ==========================================
// 4. MODEL 3: RANDOM FOREST CLASSIFIER
// ==========================================

export class RandomForest {
  private trees: DecisionTree[] = [];
  private numTrees: number;
  private maxDepth: number;
  private featureSubsampleRatio: number;

  constructor(numTrees = 10, maxDepth = 6, featureSubsampleRatio = 0.7) {
    this.numTrees = numTrees;
    this.maxDepth = maxDepth;
    this.featureSubsampleRatio = featureSubsampleRatio;
  }

  public train(data: PreprocessedFeatures[], seed = 42) {
    this.trees = [];
    const n = data.length;
    let localSeed = seed;
    const rand = () => {
      const x = Math.sin(localSeed++) * 10000;
      return x - Math.floor(x);
    };

    for (let t = 0; t < this.numTrees; t++) {
      // 1. Bootstrap Sample (sampling with replacement)
      const bootstrap: PreprocessedFeatures[] = [];
      for (let i = 0; i < n; i++) {
        const randIdx = Math.floor(rand() * n);
        bootstrap.push(data[randIdx]);
      }

      // 2. Instantiate and train a DecisionTree
      // Scikit-learn Random Forests split nodes on a random subset of features.
      // For this pure-TS engine, we can pass down parameters or train depth-limited trees on bootstraps.
      const tree = new DecisionTree(this.maxDepth, 4);
      tree.train(bootstrap);
      this.trees.push(tree);
    }
  }

  public predictProb(x: number[]): number {
    if (this.trees.length === 0) return 0.5;
    let probSum = 0;
    this.trees.forEach(tree => {
      probSum += tree.predictProb(x);
    });
    return probSum / this.trees.length;
  }

  public getFeatureImportance(featureCount: number): number[] {
    const importances = new Array(featureCount).fill(0);
    this.trees.forEach(tree => {
      const treeImportance = tree.getFeatureImportance(featureCount);
      for (let j = 0; j < featureCount; j++) {
        importances[j] += treeImportance[j];
      }
    });
    const sum = importances.reduce((a, b) => a + b, 0);
    return sum > 0 ? importances.map(v => v / sum) : importances;
  }
}


// ==========================================
// 5. EVALUATION METRICS ENGINE
// ==========================================

/**
 * Calculates model metrics on a test set
 */
export function evaluateModel(
  model: { predictProb: (x: number[]) => number; getFeatureImportance?: (fc: number) => number[] },
  testSet: PreprocessedFeatures[],
  featureNames: string[],
  name: string,
  threshold: number = 0.5
): ModelResults {
  const predictions = testSet.map(item => ({
    prob: model.predictProb(item.x),
    actual: item.y
  }));

  // Classification results
  let tp = 0; // True Positive
  let fp = 0; // False Positive
  let tn = 0; // True Negative
  let fn = 0; // False Negative

  predictions.forEach(p => {
    const predClass = p.prob >= threshold ? 1 : 0;
    if (predClass === 1 && p.actual === 1) tp++;
    else if (predClass === 1 && p.actual === 0) fp++;
    else if (predClass === 0 && p.actual === 0) tn++;
    else if (predClass === 0 && p.actual === 1) fn++;
  });

  const accuracy = (tp + tn) / testSet.length;
  const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
  const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
  const f1Score = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;

  // Compute ROC points and AUC
  // 1. Sort predictions descending by probability
  const sortedPredictions = [...predictions].sort((a, b) => b.prob - a.prob);

  const totalPositives = testSet.filter(d => d.y === 1).length;
  const totalNegatives = testSet.length - totalPositives;

  const rocCurve: ROCPoint[] = [{ fpr: 0, tpr: 0 }];
  
  let currentTp = 0;
  let currentFp = 0;
  let auc = 0;
  let prevFpr = 0;
  let prevTpr = 0;

  for (let i = 0; i < sortedPredictions.length; i++) {
    const item = sortedPredictions[i];
    if (item.actual === 1) {
      currentTp++;
    } else {
      currentFp++;
    }

    const tpr = totalPositives > 0 ? currentTp / totalPositives : 0;
    const fpr = totalNegatives > 0 ? currentFp / totalNegatives : 0;

    rocCurve.push({ fpr, tpr });

    // Trapezoidal rule integration for AUC
    if (fpr !== prevFpr) {
      auc += ((tpr + prevTpr) / 2) * (fpr - prevFpr);
      prevFpr = fpr;
      prevTpr = tpr;
    }
  }
  // Ensure we reach (1,1)
  if (rocCurve[rocCurve.length - 1].fpr !== 1 || rocCurve[rocCurve.length - 1].tpr !== 1) {
    rocCurve.push({ fpr: 1, tpr: 1 });
    auc += ((1 + prevTpr) / 2) * (1 - prevFpr);
  }

  // Compute Feature Importance
  let finalImportance: { feature: string; value: number }[] = [];
  if (model.getFeatureImportance) {
    const imp = model.getFeatureImportance(featureNames.length);
    finalImportance = featureNames.map((name, i) => ({
      feature: name,
      value: parseFloat((imp[i] || 0).toFixed(3))
    }));
  } else if (name === 'Logistic Regression') {
    // For logistic regression, use weight magnitudes as importance proxy
    const weights = (model as LogisticRegression).getWeights();
    const sumMag = weights.reduce((a, b) => a + Math.abs(b), 0);
    finalImportance = featureNames.map((fname, i) => ({
      feature: fname,
      value: sumMag > 0 ? parseFloat((Math.abs(weights[i]) / sumMag).toFixed(3)) : 0
    }));
  }

  // Sort importance descending
  finalImportance.sort((a, b) => b.value - a.value);

  return {
    name,
    metrics: {
      accuracy: parseFloat(accuracy.toFixed(3)),
      precision: parseFloat(precision.toFixed(3)),
      recall: parseFloat(recall.toFixed(3)),
      f1Score: parseFloat(f1Score.toFixed(3)),
      auc: parseFloat(auc.toFixed(3))
    },
    rocCurve,
    featureImportance: finalImportance
  };
}

/**
 * High-level function to run the full training and evaluation pipeline
 */
export function runMLExperiment(
  rawRecords: CreditDataRecord[],
  hyperparams: ModelHyperparameters,
  trainRatio: number = 0.7,
  threshold: number = 0.5
): {
  results: Record<string, ModelResults>;
  preprocessingLog: string[];
} {
  const { trainSet, testSet, featureNames, preprocessingLog } = preprocessData(rawRecords, trainRatio, true);

  // 1. Train Logistic Regression
  const lr = new LogisticRegression(
    hyperparams.logisticRegression.learningRate,
    hyperparams.logisticRegression.iterations,
    hyperparams.logisticRegression.regularization,
    hyperparams.logisticRegression.lambda
  );
  lr.train(trainSet, featureNames.length);
  const lrEval = evaluateModel(lr, testSet, featureNames, 'Logistic Regression', threshold);

  // 2. Train Decision Tree
  const dt = new DecisionTree(
    hyperparams.decisionTree.maxDepth,
    hyperparams.decisionTree.minSamplesSplit
  );
  dt.train(trainSet);
  const dtEval = evaluateModel(dt, testSet, featureNames, 'Decision Tree', threshold);

  // 3. Train Random Forest
  const rf = new RandomForest(
    hyperparams.randomForest.numTrees,
    hyperparams.randomForest.maxDepth,
    hyperparams.randomForest.featureSubsampleRatio
  );
  rf.train(trainSet, 42); // standard seed
  const rfEval = evaluateModel(rf, testSet, featureNames, 'Random Forest', threshold);

  return {
    results: {
      logisticRegression: lrEval,
      decisionTree: dtEval,
      randomForest: rfEval
    },
    preprocessingLog
  };
}
