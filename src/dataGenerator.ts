import { CreditDataRecord, DatasetScenario } from './types';

export const SCENARIOS: DatasetScenario[] = [
  {
    id: 'balanced',
    name: 'Balanced Standard',
    description: 'A standard economic climate with a balanced mix of credit profiles and realistic default rates.',
    avgIncome: 65,      // Mean income of $65k
    defaultRate: 0.25,  // ~25% bad credit risk
    noiseLevel: 0.8,
  },
  {
    id: 'recession',
    name: 'Economic Recession',
    description: 'Simulates a high-stress economic downturn. Lower average incomes, higher debt ratios, and elevated credit risk.',
    avgIncome: 45,      // Mean income of $45k
    defaultRate: 0.45,  // ~45% bad credit risk
    noiseLevel: 1.2,
  },
  {
    id: 'boom',
    name: 'Economic Boom',
    description: 'Simulates a robust, high-growth economy. Stronger incomes, lower debt levels, and highly favorable credit scores.',
    avgIncome: 85,      // Mean income of $85k
    defaultRate: 0.12,  // ~12% bad credit risk
    noiseLevel: 0.5,
  }
];

// Helper to generate pseudo-random numbers with a seed for deterministic behavior
export function seedRandom(seed: number) {
  let localSeed = seed;
  return function() {
    const x = Math.sin(localSeed++) * 10000;
    return x - Math.floor(x);
  };
}

// Generate Box-Muller transform for normal distribution
export function randomNormal(mean: number, stddev: number, randFn: () => number) {
  const u1 = randFn() || 0.0001; // Avoid 0
  const u2 = randFn() || 0.0001;
  const randStdNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return mean + stddev * randStdNormal;
}

export function generateCreditDataset(scenario: DatasetScenario, size: number = 250, seed: number = 42): CreditDataRecord[] {
  const rand = seedRandom(seed);
  const records: CreditDataRecord[] = [];

  for (let i = 0; i < size; i++) {
    // 1. Employment Status
    const empRand = rand();
    let employment_status: 'Unemployed' | 'Employed' | 'Self-Employed' = 'Employed';
    let empScore = 0; // influence on creditworthiness
    if (empRand < 0.12) {
      employment_status = 'Unemployed';
      empScore = -2.5;
    } else if (empRand < 0.30) {
      employment_status = 'Self-Employed';
      empScore = 0.5;
    } else {
      employment_status = 'Employed';
      empScore = 2.0;
    }

    // 2. Age (Normally distributed around 40, bounded 18 to 75)
    let age = Math.round(randomNormal(42, 13, rand));
    if (age < 18) age = 18;
    if (age > 75) age = 75;

    // 3. Income (Log-normal distribution simulated)
    const baseIncome = scenario.avgIncome;
    let income = Math.round(randomNormal(baseIncome, baseIncome * 0.35, rand));
    if (income < 15) income = 15; // minimum income $15k

    // 4. Debt-to-income Ratio in % (0 to 100%)
    // Typically higher for lower incomes
    const baseDebt = scenario.id === 'recession' ? 48 : scenario.id === 'boom' ? 25 : 35;
    let debt = Math.round(randomNormal(baseDebt + (100 / (income / 10)), 15, rand));
    if (debt < 5) debt = 5;
    if (debt > 95) debt = 95;

    // 5. Payment History Score (0 to 100, where 100 is excellent)
    const baseHistory = scenario.id === 'recession' ? 68 : scenario.id === 'boom' ? 88 : 80;
    // Payment history is strongly linked to employment and age (older people tend to have better history)
    const ageBonus = (age - 18) * 0.3;
    const debtPenalty = (debt - 30) * 0.4;
    let payment_history = Math.round(randomNormal(baseHistory + ageBonus - debtPenalty, 12, rand));
    if (payment_history < 10) payment_history = 10;
    if (payment_history > 100) payment_history = 100;

    // 6. Loan Amount (in thousands)
    // Linked to income (higher income gets larger loans)
    let loan_amount = Math.round(income * randomNormal(0.6, 0.15, rand));
    if (loan_amount < 3) loan_amount = 3;
    if (loan_amount > income * 1.5) loan_amount = Math.round(income * 1.5);

    // 7. Calculate Credit Risk (0 = Bad, 1 = Good)
    // Score components scaled to a standard logistic function
    // Z = Beta_0 + sum(Beta_i * X_i)
    const intercept = -3.5 + (scenario.id === 'recession' ? -1.0 : scenario.id === 'boom' ? 1.5 : 0);
    const w_income = 0.035;       // higher income is good
    const w_age = 0.02;          // older age is good
    const w_debt = -0.055;        // higher debt ratio is bad
    const w_history = 0.065;      // higher payment score is extremely good
    const w_loan = -0.015;        // higher loan amount increases default probability
    const w_emp = 0.4 * empScore; // employment status weight

    const z = intercept + 
              (income * w_income) + 
              (age * w_age) + 
              (debt * w_debt) + 
              (payment_history * w_history) + 
              (loan_amount * w_loan) + 
              w_emp + 
              randomNormal(0, scenario.noiseLevel, rand);

    const prob = 1 / (1 + Math.exp(-z));
    const credit_risk: 0 | 1 = prob >= 0.5 ? 1 : 0;

    records.push({
      id: i + 1,
      income,
      age,
      debt,
      payment_history,
      loan_amount,
      employment_status,
      credit_risk
    });
  }

  return records;
}

// Compute Pearson correlation matrix between numerical features and credit_risk
export function calculateCorrelationMatrix(data: CreditDataRecord[]) {
  const fields = ['income', 'age', 'debt', 'payment_history', 'loan_amount', 'credit_risk'];
  const n = data.length;
  
  const means: Record<string, number> = {};
  fields.forEach(f => {
    means[f] = data.reduce((sum, item) => sum + (item[f as keyof CreditDataRecord] as number), 0) / n;
  });

  const matrix: Record<string, Record<string, number>> = {};
  
  fields.forEach(f1 => {
    matrix[f1] = {};
    fields.forEach(f2 => {
      let num = 0;
      let den1 = 0;
      let den2 = 0;
      
      data.forEach(item => {
        const val1 = item[f1 as keyof CreditDataRecord] as number;
        const val2 = item[f2 as keyof CreditDataRecord] as number;
        const diff1 = val1 - means[f1];
        const diff2 = val2 - means[f2];
        
        num += diff1 * diff2;
        den1 += diff1 * diff1;
        den2 += diff2 * diff2;
      });
      
      const r = den1 && den2 ? num / Math.sqrt(den1 * den2) : 0;
      matrix[f1][f2] = parseFloat(r.toFixed(3));
    });
  });

  return matrix;
}

// Compute general summary statistics
export function calculateSummaryStats(data: CreditDataRecord[]) {
  const size = data.length;
  const goodRiskCount = data.filter(d => d.credit_risk === 1).length;
  const badRiskCount = size - goodRiskCount;
  
  const sumOf = (f: keyof CreditDataRecord) => data.reduce((sum, item) => sum + (item[f] as number), 0);
  
  return {
    total: size,
    goodRisk: goodRiskCount,
    goodRiskPercent: parseFloat(((goodRiskCount / size) * 100).toFixed(1)),
    badRisk: badRiskCount,
    badRiskPercent: parseFloat(((badRiskCount / size) * 100).toFixed(1)),
    avgIncome: parseFloat((sumOf('income') / size).toFixed(1)),
    avgAge: parseFloat((sumOf('age') / size).toFixed(1)),
    avgDebt: parseFloat((sumOf('debt') / size).toFixed(1)),
    avgHistory: parseFloat((sumOf('payment_history') / size).toFixed(1)),
    avgLoan: parseFloat((sumOf('loan_amount') / size).toFixed(1)),
  };
}
