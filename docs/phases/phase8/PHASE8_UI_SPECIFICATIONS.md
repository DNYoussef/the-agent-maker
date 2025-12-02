# Phase 8: UI Component Specifications

**Version**: 2.0
**Status**: Complete Design
**Purpose**: Real-time monitoring and visualization of Phase 8 compression with quality validation

---

## Executive Summary

Phase 8 UI provides comprehensive monitoring of the 3-stage compression pipeline (SeedLM → VPTQ → Hypercompression) with real-time benchmark testing visualization. The dashboard enables users to track compression progress, quality degradation, and automatic rollback events.

**User Requirement**: "has a ui component" for Phase 8 compression monitoring ✅

---

## 1. Dashboard Architecture

### 1.1 Overall Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  Phase 8: Final Compression Dashboard                           │
├──────────────────────────────────────────────────────────────────┤
│  [Baseline] → [SeedLM] → [VPTQ] → [Hypercompression]           │
│     100MB       50MB       2.5MB        0.4MB                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  CURRENT STAGE: VPTQ Compression                        │   │
│  │  Progress: ████████████░░░░░░ 67% (2.3 hrs elapsed)     │   │
│  │  Status: Running benchmark tests...                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Compression     │  │ Quality Metrics │  │ Benchmark       │ │
│  │ Progress        │  │                 │  │ Results         │ │
│  │                 │  │                 │  │                 │ │
│  │ [Chart]         │  │ [Gauges]        │  │ [Table]         │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Quality vs Compression Tradeoff                        │   │
│  │  [Interactive Line Chart]                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────────────────────────┐  │
│  │ Expert-Specific │  │ Integration Tests                   │  │
│  │ Performance     │  │ • Edge-of-Chaos: 74.8% ✅           │  │
│  │ [Radar Chart]   │  │ • Eudaimonia: All rules passed ✅   │  │
│  └─────────────────┘  └─────────────────────────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Event Log & Alerts                                     │   │
│  │  14:23:45 ⚠️  VPTQ quality gate failed (MMLU: 89%)     │   │
│  │  14:23:50 🔄 Retrying VPTQ with adjusted hyperparams    │   │
│  │  14:35:12 ✅ VPTQ retry passed quality gate             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Hierarchy

```typescript
<Phase8Dashboard>
  <ProgressTimeline />
  <CurrentStageCard />
  <Grid>
    <CompressionProgressChart />
    <QualityMetrics />
    <BenchmarkResultsTable />
  </Grid>
  <QualityVsCompressionChart />
  <Grid>
    <ExpertPerformanceRadar />
    <IntegrationTestsPanel />
  </Grid>
  <EventLogPanel />
  <ControlPanel />
</Phase8Dashboard>
```

---

## 2. Core Components

### 2.1 Progress Timeline

**Purpose**: Visual pipeline showing compression stages and current progress

```typescript
interface ProgressTimelineProps {
  currentStage: 'baseline' | 'seedlm' | 'vptq' | 'hypercompression' | 'complete';
  stageStatuses: {
    baseline: 'complete' | 'in_progress' | 'pending';
    seedlm: 'complete' | 'in_progress' | 'pending' | 'failed' | 'retrying';
    vptq: 'complete' | 'in_progress' | 'pending' | 'failed' | 'retrying';
    hypercompression: 'complete' | 'in_progress' | 'pending' | 'failed' | 'retrying';
  };
  modelSizes: {
    baseline: number;    // 100MB
    seedlm: number;      // 50MB
    vptq: number;        // 2.5MB
    hypercompression: number;  // 0.4MB
  };
}

const ProgressTimeline: React.FC<ProgressTimelineProps> = ({
  currentStage,
  stageStatuses,
  modelSizes
}) => {
  return (
    <div className="flex items-center justify-between p-6 bg-gray-50 rounded-lg">
      {/* Baseline */}
      <StageNode
        label="Baseline"
        size={`${modelSizes.baseline}MB`}
        status={stageStatuses.baseline}
        isActive={currentStage === 'baseline'}
      />
      <Arrow />

      {/* SeedLM */}
      <StageNode
        label="SeedLM"
        size={`${modelSizes.seedlm}MB`}
        compression="2×"
        status={stageStatuses.seedlm}
        isActive={currentStage === 'seedlm'}
      />
      <Arrow />

      {/* VPTQ */}
      <StageNode
        label="VPTQ"
        size={`${modelSizes.vptq}MB`}
        compression="20×"
        status={stageStatuses.vptq}
        isActive={currentStage === 'vptq'}
      />
      <Arrow />

      {/* Hypercompression */}
      <StageNode
        label="Hypercompression"
        size={`${modelSizes.hypercompression}MB`}
        compression="6.25×"
        status={stageStatuses.hypercompression}
        isActive={currentStage === 'hypercompression'}
      />
    </div>
  );
};
```

**Visual States**:
- ✅ **Complete**: Green checkmark, solid border
- 🔄 **In Progress**: Blue pulsing animation
- ⏳ **Pending**: Gray, dotted border
- ❌ **Failed**: Red X icon
- 🔄 **Retrying**: Orange pulsing animation

---

### 2.2 Current Stage Card

**Purpose**: Detailed view of current compression/testing stage

```typescript
interface CurrentStageCardProps {
  stage: CompressionStage;
  progress: number;  // 0-100
  elapsedTime: number;  // seconds
  currentActivity: string;
  logs: string[];
}

const CurrentStageCard: React.FC<CurrentStageCardProps> = ({
  stage,
  progress,
  elapsedTime,
  currentActivity,
  logs
}) => {
  return (
    <div className="p-6 bg-white rounded-lg shadow-md border-2 border-blue-500">
      <h3 className="text-2xl font-bold mb-4">
        {stage.name} Compression
      </h3>

      {/* Progress bar */}
      <div className="mb-4">
        <div className="flex justify-between mb-2">
          <span>Progress: {progress}%</span>
          <span>Elapsed: {formatTime(elapsedTime)}</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-4">
          <div
            className="bg-blue-600 h-4 rounded-full transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Current activity */}
      <div className="mb-4 p-3 bg-blue-50 rounded">
        <p className="text-sm font-medium">
          {currentActivity}
        </p>
      </div>

      {/* Recent logs (last 5) */}
      <div className="text-xs font-mono text-gray-600">
        {logs.slice(-5).map((log, i) => (
          <div key={i} className="py-1">{log}</div>
        ))}
      </div>
    </div>
  );
};
```

**Activity Examples**:
- "Applying SeedLM compression... (Step 2/5)"
- "Running MMLU benchmark... (312/500 samples)"
- "Validating quality gates..."
- "⚠️ Quality threshold failed. Preparing retry..."
- "✅ Quality gate passed. Proceeding to next stage."

---

### 2.3 Compression Progress Chart

**Purpose**: Visualize model size reduction over time

```typescript
interface CompressionProgressChartProps {
  data: Array<{
    stage: string;
    size_mb: number;
    compression_ratio: number;
    timestamp: Date;
  }>;
}

const CompressionProgressChart: React.FC<CompressionProgressChartProps> = ({ data }) => {
  // Using Recharts library
  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <h4 className="text-lg font-semibold mb-4">Compression Progress</h4>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="stage" />
          <YAxis
            label={{ value: 'Model Size (MB)', angle: -90, position: 'insideLeft' }}
            scale="log"
          />
          <Tooltip />
          <Legend />
          <Line
            type="monotone"
            dataKey="size_mb"
            stroke="#3b82f6"
            strokeWidth={3}
            dot={{ r: 6 }}
          />
        </LineChart>
      </ResponsiveContainer>

      {/* Compression ratio annotations */}
      <div className="mt-4 flex justify-around text-sm">
        {data.map((point) => (
          <div key={point.stage} className="text-center">
            <div className="font-bold text-blue-600">
              {point.compression_ratio}×
            </div>
            <div className="text-gray-500">{point.stage}</div>
          </div>
        ))}
      </div>
    </div>
  );
};
```

**Chart Features**:
- Logarithmic Y-axis (100MB → 0.4MB range)
- Animated line drawing
- Hover tooltips with compression ratio
- Milestone markers at each stage

---

### 2.4 Quality Metrics Dashboard

**Purpose**: Real-time display of quality retention metrics

```typescript
interface QualityMetricsProps {
  currentMetrics: {
    mmlu: number;
    gsm8k: number;
    humaneval: number;
    hellaswag: number;
    overall_retention: number;
  };
  baselineMetrics: {
    mmlu: number;
    gsm8k: number;
    humaneval: number;
    hellaswag: number;
  };
  thresholds: {
    core_benchmark: number;  // 0.95
    overall: number;         // 0.84
  };
}

const QualityMetrics: React.FC<QualityMetricsProps> = ({
  currentMetrics,
  baselineMetrics,
  thresholds
}) => {
  const retention = {
    mmlu: currentMetrics.mmlu / baselineMetrics.mmlu,
    gsm8k: currentMetrics.gsm8k / baselineMetrics.gsm8k,
    humaneval: currentMetrics.humaneval / baselineMetrics.humaneval,
    hellaswag: currentMetrics.hellaswag / baselineMetrics.hellaswag
  };

  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <h4 className="text-lg font-semibold mb-4">Quality Metrics</h4>

      {/* Overall retention gauge */}
      <div className="mb-6 flex justify-center">
        <CircularGauge
          value={currentMetrics.overall_retention}
          max={1.0}
          threshold={thresholds.overall}
          label="Overall Retention"
          size={200}
        />
      </div>

      {/* Individual benchmark retention */}
      <div className="grid grid-cols-2 gap-4">
        <MetricCard
          name="MMLU"
          current={currentMetrics.mmlu}
          baseline={baselineMetrics.mmlu}
          retention={retention.mmlu}
          threshold={thresholds.core_benchmark}
        />
        <MetricCard
          name="GSM8K"
          current={currentMetrics.gsm8k}
          baseline={baselineMetrics.gsm8k}
          retention={retention.gsm8k}
          threshold={thresholds.core_benchmark}
        />
        <MetricCard
          name="HumanEval"
          current={currentMetrics.humaneval}
          baseline={baselineMetrics.humaneval}
          retention={retention.humaneval}
          threshold={thresholds.core_benchmark}
        />
        <MetricCard
          name="HellaSwag"
          current={currentMetrics.hellaswag}
          baseline={baselineMetrics.hellaswag}
          retention={retention.hellaswag}
          threshold={thresholds.core_benchmark}
        />
      </div>
    </div>
  );
};

// Metric card component
const MetricCard = ({ name, current, baseline, retention, threshold }) => {
  const passed = retention >= threshold;

  return (
    <div className={`p-3 rounded border-2 ${
      passed ? 'border-green-500 bg-green-50' : 'border-red-500 bg-red-50'
    }`}>
      <div className="flex items-center justify-between mb-2">
        <span className="font-semibold">{name}</span>
        {passed ? <CheckIcon className="text-green-600" /> : <XIcon className="text-red-600" />}
      </div>
      <div className="text-2xl font-bold mb-1">
        {(retention * 100).toFixed(1)}%
      </div>
      <div className="text-sm text-gray-600">
        {current.toFixed(3)} / {baseline.toFixed(3)}
      </div>
    </div>
  );
};
```

**Circular Gauge Specifications**:
- Green zone: ≥95% retention
- Yellow zone: 84-95% retention
- Red zone: <84% retention
- Animated needle transition
- Threshold marker line

---

### 2.5 Benchmark Results Table

**Purpose**: Detailed comparison table of all benchmark scores

```typescript
interface BenchmarkResultsTableProps {
  results: Array<{
    benchmark: string;
    baseline: number;
    seedlm: number | null;
    vptq: number | null;
    hypercompression: number | null;
    threshold: number;
  }>;
  currentStage: string;
}

const BenchmarkResultsTable: React.FC<BenchmarkResultsTableProps> = ({
  results,
  currentStage
}) => {
  return (
    <div className="p-4 bg-white rounded-lg shadow overflow-x-auto">
      <h4 className="text-lg font-semibold mb-4">Benchmark Comparison</h4>

      <table className="w-full text-sm">
        <thead>
          <tr className="border-b-2 border-gray-300">
            <th className="text-left py-2 px-3">Benchmark</th>
            <th className="text-right py-2 px-3">Baseline</th>
            <th className="text-right py-2 px-3">SeedLM</th>
            <th className="text-right py-2 px-3">VPTQ</th>
            <th className="text-right py-2 px-3">Hyper</th>
            <th className="text-right py-2 px-3">Retention</th>
            <th className="text-center py-2 px-3">Status</th>
          </tr>
        </thead>
        <tbody>
          {results.map((row) => {
            const finalScore = row.hypercompression ?? row.vptq ?? row.seedlm;
            const retention = finalScore ? finalScore / row.baseline : null;
            const passed = retention ? retention >= row.threshold : null;

            return (
              <tr key={row.benchmark} className="border-b border-gray-200 hover:bg-gray-50">
                <td className="py-2 px-3 font-medium">{row.benchmark}</td>
                <td className="py-2 px-3 text-right">{row.baseline.toFixed(3)}</td>
                <td className="py-2 px-3 text-right">
                  {row.seedlm ? row.seedlm.toFixed(3) : '-'}
                </td>
                <td className="py-2 px-3 text-right">
                  {row.vptq ? row.vptq.toFixed(3) : '-'}
                </td>
                <td className="py-2 px-3 text-right">
                  {row.hypercompression ? row.hypercompression.toFixed(3) : '-'}
                </td>
                <td className="py-2 px-3 text-right">
                  {retention ? (
                    <span className={passed ? 'text-green-600' : 'text-red-600'}>
                      {(retention * 100).toFixed(1)}%
                    </span>
                  ) : '-'}
                </td>
                <td className="py-2 px-3 text-center">
                  {passed === null ? '⏳' : passed ? '✅' : '❌'}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {/* Summary row */}
      <div className="mt-4 p-3 bg-gray-100 rounded text-sm">
        <span className="font-semibold">Summary:</span> {' '}
        {results.filter(r => {
          const finalScore = r.hypercompression ?? r.vptq ?? r.seedlm;
          const retention = finalScore ? finalScore / r.baseline : null;
          return retention && retention >= r.threshold;
        }).length} / {results.length} benchmarks passed quality gates
      </div>
    </div>
  );
};
```

**Table Features**:
- Sortable columns
- Color-coded retention percentages
- Expandable rows (show sub-benchmarks, e.g., MMLU subjects)
- Export to CSV button

---

### 2.6 Quality vs Compression Tradeoff Chart

**Purpose**: Interactive visualization of quality degradation vs compression ratio

```typescript
interface QualityVsCompressionChartProps {
  data: Array<{
    stage: string;
    compression_ratio: number;
    mmlu: number;
    gsm8k: number;
    humaneval: number;
    hellaswag: number;
  }>;
}

const QualityVsCompressionChart: React.FC<QualityVsCompressionChartProps> = ({ data }) => {
  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <h4 className="text-lg font-semibold mb-4">Quality vs Compression Tradeoff</h4>

      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="compression_ratio"
            label={{ value: 'Compression Ratio', position: 'insideBottom', offset: -5 }}
            scale="log"
          />
          <YAxis
            label={{ value: 'Accuracy', angle: -90, position: 'insideLeft' }}
            domain={[0.4, 0.9]}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />

          {/* Multiple lines for different benchmarks */}
          <Line
            type="monotone"
            dataKey="mmlu"
            stroke="#3b82f6"
            strokeWidth={2}
            name="MMLU"
            dot={{ r: 4 }}
          />
          <Line
            type="monotone"
            dataKey="gsm8k"
            stroke="#10b981"
            strokeWidth={2}
            name="GSM8K"
            dot={{ r: 4 }}
          />
          <Line
            type="monotone"
            dataKey="humaneval"
            stroke="#f59e0b"
            strokeWidth={2}
            name="HumanEval"
            dot={{ r: 4 }}
          />
          <Line
            type="monotone"
            dataKey="hellaswag"
            stroke="#8b5cf6"
            strokeWidth={2}
            name="HellaSwag"
            dot={{ r: 4 }}
          />

          {/* Threshold line at 84% */}
          <ReferenceLine
            y={0.84}
            stroke="red"
            strokeDasharray="3 3"
            label="Min Threshold (84%)"
          />
        </LineChart>
      </ResponsiveContainer>

      {/* Interactive toggles */}
      <div className="mt-4 flex gap-4 justify-center">
        <button className="px-4 py-2 bg-blue-100 rounded hover:bg-blue-200">
          Toggle MMLU
        </button>
        <button className="px-4 py-2 bg-green-100 rounded hover:bg-green-200">
          Toggle GSM8K
        </button>
        <button className="px-4 py-2 bg-orange-100 rounded hover:bg-orange-200">
          Toggle HumanEval
        </button>
        <button className="px-4 py-2 bg-purple-100 rounded hover:bg-purple-200">
          Toggle HellaSwag
        </button>
      </div>
    </div>
  );
};
```

**Chart Insights**:
- X-axis: Logarithmic compression ratio (1× → 280×)
- Y-axis: Accuracy (0.4 to 0.9)
- Threshold line at 84% (user requirement: "dont lose to much quality")
- Hover shows exact values and retention percentage

---

### 2.7 Expert Performance Radar Chart

**Purpose**: Visualize expert-specific benchmark performance

```typescript
interface ExpertPerformanceRadarProps {
  experts: Array<{
    name: string;
    benchmarks: {
      [key: string]: {
        baseline: number;
        current: number;
      };
    };
  }>;
}

const ExpertPerformanceRadar: React.FC<ExpertPerformanceRadarProps> = ({ experts }) => {
  // Transform data for radar chart
  const radarData = experts[0].benchmarks.map((benchmark, i) => {
    const dataPoint: any = { benchmark: benchmark.name };

    experts.forEach(expert => {
      const retention = expert.benchmarks[i].current / expert.benchmarks[i].baseline;
      dataPoint[expert.name] = retention;
    });

    return dataPoint;
  });

  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <h4 className="text-lg font-semibold mb-4">Expert-Specific Performance</h4>

      <ResponsiveContainer width="100%" height={350}>
        <RadarChart data={radarData}>
          <PolarGrid />
          <PolarAngleAxis dataKey="benchmark" />
          <PolarRadiusAxis domain={[0, 1]} />
          <Tooltip />
          <Legend />

          {experts.map((expert, i) => (
            <Radar
              key={expert.name}
              name={expert.name}
              dataKey={expert.name}
              stroke={EXPERT_COLORS[i]}
              fill={EXPERT_COLORS[i]}
              fillOpacity={0.3}
            />
          ))}

          {/* Threshold circle at 95% retention */}
          <PolarRadiusAxis
            domain={[0, 1]}
            tick={false}
          >
            <circle cx="50%" cy="50%" r="48%" fill="none" stroke="red" strokeDasharray="5 5" />
          </PolarRadiusAxis>
        </RadarChart>
      </ResponsiveContainer>

      {/* Expert legend with color coding */}
      <div className="mt-4 grid grid-cols-3 gap-2">
        {experts.map((expert, i) => (
          <div key={expert.name} className="flex items-center gap-2">
            <div
              className="w-4 h-4 rounded"
              style={{ backgroundColor: EXPERT_COLORS[i] }}
            />
            <span className="text-sm">{expert.name}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
```

**Expert Colors**:
```typescript
const EXPERT_COLORS = [
  '#3b82f6',  // Blue (analytical)
  '#10b981',  // Green (creative)
  '#f59e0b',  // Orange (code)
  '#8b5cf6',  // Purple (reasoning)
  '#ef4444',  // Red (communication)
  '#06b6d4',  // Cyan (memory)
  '#f97316',  // Deep orange (planning)
  '#84cc16',  // Lime (execution)
];
```

---

### 2.8 Integration Tests Panel

**Purpose**: Display Phase 5 integration test results (edge-of-chaos, eudaimonia)

```typescript
interface IntegrationTestsPanelProps {
  edgeOfChaos: {
    accuracy: number;
    status: 'optimal' | 'too_easy' | 'too_hard';
    target_min: number;  // 0.70
    target_max: number;  // 0.80
  };
  eudaimonia: {
    autonomy: number;
    honesty: number;
    harm: number;
    dignity: number;
    threshold: number;  // 0.65 (or 0.60 for hypercompression)
  };
}

const IntegrationTestsPanel: React.FC<IntegrationTestsPanelProps> = ({
  edgeOfChaos,
  eudaimonia
}) => {
  const eocPassed = edgeOfChaos.status === 'optimal';
  const eudPassed = Object.values(eudaimonia).every(
    (v, i) => i === 4 || v >= eudaimonia.threshold  // Skip 'threshold' key
  );

  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <h4 className="text-lg font-semibold mb-4">Integration Tests (Phase 5)</h4>

      {/* Edge-of-Chaos */}
      <div className={`mb-4 p-4 rounded border-2 ${
        eocPassed ? 'border-green-500 bg-green-50' : 'border-red-500 bg-red-50'
      }`}>
        <div className="flex items-center justify-between mb-2">
          <h5 className="font-semibold">Edge-of-Chaos Learning Zone</h5>
          {eocPassed ? <CheckIcon className="text-green-600" /> : <XIcon className="text-red-600" />}
        </div>

        <div className="mb-2">
          <div className="flex justify-between text-sm mb-1">
            <span>Accuracy: {(edgeOfChaos.accuracy * 100).toFixed(1)}%</span>
            <span className="text-gray-600">Target: 70-80%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3 relative">
            {/* Target zone (70-80%) */}
            <div
              className="absolute bg-green-200 h-3 rounded-full"
              style={{
                left: '70%',
                width: '10%'
              }}
            />
            {/* Current accuracy */}
            <div
              className={`h-3 rounded-full ${
                eocPassed ? 'bg-green-600' : 'bg-red-600'
              }`}
              style={{ width: `${edgeOfChaos.accuracy * 100}%` }}
            />
          </div>
        </div>

        <div className="text-sm text-gray-700">
          Status: {edgeOfChaos.status === 'optimal' ? '✅ Optimal learning zone' :
                   edgeOfChaos.status === 'too_easy' ? '⚠️ Too easy - may lose generalization' :
                   '⚠️ Too difficult - lost core capability'}
        </div>
      </div>

      {/* Eudaimonia */}
      <div className={`p-4 rounded border-2 ${
        eudPassed ? 'border-green-500 bg-green-50' : 'border-red-500 bg-red-50'
      }`}>
        <div className="flex items-center justify-between mb-3">
          <h5 className="font-semibold">Eudaimonia Moral Alignment</h5>
          {eudPassed ? <CheckIcon className="text-green-600" /> : <XIcon className="text-red-600" />}
        </div>

        <div className="grid grid-cols-2 gap-3">
          <EudaimoniaRule
            name="Autonomy"
            score={eudaimonia.autonomy}
            threshold={eudaimonia.threshold}
          />
          <EudaimoniaRule
            name="Honesty"
            score={eudaimonia.honesty}
            threshold={eudaimonia.threshold}
          />
          <EudaimoniaRule
            name="Minimize Harm"
            score={eudaimonia.harm}
            threshold={eudaimonia.threshold}
          />
          <EudaimoniaRule
            name="Dignity"
            score={eudaimonia.dignity}
            threshold={eudaimonia.threshold}
          />
        </div>

        <div className="mt-3 text-sm text-gray-700">
          All rules must maintain ≥{(eudaimonia.threshold * 100).toFixed(0)}% for quality gate pass
        </div>
      </div>
    </div>
  );
};

// Eudaimonia rule mini-card
const EudaimoniaRule = ({ name, score, threshold }) => {
  const passed = score >= threshold;

  return (
    <div className="flex items-center justify-between p-2 bg-white rounded border">
      <span className="text-sm font-medium">{name}</span>
      <div className="flex items-center gap-2">
        <span className={`font-bold ${passed ? 'text-green-600' : 'text-red-600'}`}>
          {(score * 100).toFixed(0)}%
        </span>
        {passed ? <CheckIcon className="w-4 h-4 text-green-600" /> : <XIcon className="w-4 h-4 text-red-600" />}
      </div>
    </div>
  );
};
```

---

### 2.9 Event Log & Alerts Panel

**Purpose**: Real-time log of compression events, quality gates, retries

```typescript
interface Event {
  timestamp: Date;
  level: 'info' | 'warning' | 'error' | 'success';
  stage: string;
  message: string;
  details?: any;
}

interface EventLogPanelProps {
  events: Event[];
  autoScroll: boolean;
}

const EventLogPanel: React.FC<EventLogPanelProps> = ({ events, autoScroll }) => {
  const logRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    if (autoScroll && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [events, autoScroll]);

  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-lg font-semibold">Event Log & Alerts</h4>
        <button className="text-sm text-blue-600 hover:underline">
          Clear Log
        </button>
      </div>

      <div
        ref={logRef}
        className="h-64 overflow-y-auto font-mono text-sm border border-gray-300 rounded p-3 bg-gray-50"
      >
        {events.map((event, i) => (
          <div key={i} className={`mb-2 ${getEventColor(event.level)}`}>
            <span className="text-gray-500">
              {formatTimestamp(event.timestamp)}
            </span>
            {' '}
            <span className="font-semibold">[{event.stage.toUpperCase()}]</span>
            {' '}
            {getEventIcon(event.level)}
            {' '}
            {event.message}

            {/* Expandable details */}
            {event.details && (
              <details className="ml-6 mt-1 text-xs">
                <summary className="cursor-pointer text-blue-600">
                  Show details
                </summary>
                <pre className="mt-1 p-2 bg-white rounded border">
                  {JSON.stringify(event.details, null, 2)}
                </pre>
              </details>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

const getEventColor = (level: string) => {
  switch (level) {
    case 'info': return 'text-gray-700';
    case 'warning': return 'text-orange-600';
    case 'error': return 'text-red-600';
    case 'success': return 'text-green-600';
    default: return 'text-gray-700';
  }
};

const getEventIcon = (level: string) => {
  switch (level) {
    case 'info': return 'ℹ️';
    case 'warning': return '⚠️';
    case 'error': return '❌';
    case 'success': return '✅';
    default: return '';
  }
};
```

**Example Events**:
```typescript
const exampleEvents: Event[] = [
  {
    timestamp: new Date('2025-01-15T14:00:00'),
    level: 'info',
    stage: 'baseline',
    message: 'Establishing baseline performance...'
  },
  {
    timestamp: new Date('2025-01-15T14:15:23'),
    level: 'success',
    stage: 'baseline',
    message: 'Baseline complete. MMLU: 0.757, GSM8K: 0.685'
  },
  {
    timestamp: new Date('2025-01-15T14:16:00'),
    level: 'info',
    stage: 'seedlm',
    message: 'Starting SeedLM compression (100MB → 50MB, 2×)...'
  },
  {
    timestamp: new Date('2025-01-15T16:23:45'),
    level: 'warning',
    stage: 'vptq',
    message: 'VPTQ quality gate failed (MMLU: 0.674, 89% retention)',
    details: {
      failed_benchmarks: ['mmlu', 'gsm8k'],
      recommendation: 'Increase calibration samples by 20%'
    }
  },
  {
    timestamp: new Date('2025-01-15T16:23:50'),
    level: 'info',
    stage: 'vptq',
    message: '🔄 Retrying VPTQ with adjusted hyperparameters...'
  },
  {
    timestamp: new Date('2025-01-15T17:35:12'),
    level: 'success',
    stage: 'vptq',
    message: '✅ VPTQ retry passed quality gate'
  }
];
```

---

### 2.10 Control Panel

**Purpose**: User controls for Phase 8 execution

```typescript
interface ControlPanelProps {
  isRunning: boolean;
  isPaused: boolean;
  onStart: () => void;
  onPause: () => void;
  onResume: () => void;
  onAbort: () => void;
  onExportResults: () => void;
  settings: CompressionSettings;
  onSettingsChange: (settings: CompressionSettings) => void;
}

const ControlPanel: React.FC<ControlPanelProps> = ({
  isRunning,
  isPaused,
  onStart,
  onPause,
  onResume,
  onAbort,
  onExportResults,
  settings,
  onSettingsChange
}) => {
  return (
    <div className="fixed bottom-0 left-0 right-0 bg-white border-t-2 border-gray-300 p-4 shadow-lg">
      <div className="flex items-center justify-between max-w-7xl mx-auto">
        {/* Control buttons */}
        <div className="flex gap-3">
          {!isRunning ? (
            <button
              onClick={onStart}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-semibold"
            >
              Start Phase 8
            </button>
          ) : isPaused ? (
            <button
              onClick={onResume}
              className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 font-semibold"
            >
              Resume
            </button>
          ) : (
            <button
              onClick={onPause}
              className="px-6 py-3 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 font-semibold"
            >
              Pause
            </button>
          )}

          <button
            onClick={onAbort}
            disabled={!isRunning}
            className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:bg-gray-400 font-semibold"
          >
            Abort
          </button>

          <button
            onClick={onExportResults}
            className="px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 font-semibold"
          >
            Export Results
          </button>
        </div>

        {/* Settings button */}
        <button
          onClick={() => setShowSettings(true)}
          className="px-4 py-3 border-2 border-gray-400 rounded-lg hover:bg-gray-100"
        >
          ⚙️ Settings
        </button>
      </div>
    </div>
  );
};
```

**Settings Modal**:
```typescript
interface CompressionSettings {
  auto_retry: boolean;
  max_retries: number;
  quality_thresholds: {
    core_benchmark: number;  // Default: 0.95
    overall: number;         // Default: 0.84
  };
  fallback_strategy: 'auto' | 'manual';
  enable_hypercompression: boolean;
}

const SettingsModal = ({ settings, onSave, onClose }) => {
  return (
    <Modal isOpen onClose={onClose}>
      <h3 className="text-2xl font-bold mb-4">Compression Settings</h3>

      <div className="space-y-4">
        {/* Auto-retry */}
        <div className="flex items-center justify-between">
          <label className="font-medium">Automatic Retry on Failure</label>
          <input
            type="checkbox"
            checked={settings.auto_retry}
            onChange={(e) => updateSetting('auto_retry', e.target.checked)}
          />
        </div>

        {/* Max retries */}
        <div>
          <label className="font-medium">Max Retries per Stage</label>
          <input
            type="number"
            min="0"
            max="5"
            value={settings.max_retries}
            onChange={(e) => updateSetting('max_retries', parseInt(e.target.value))}
            className="ml-4 px-3 py-2 border rounded"
          />
        </div>

        {/* Quality thresholds */}
        <div>
          <label className="font-medium">Core Benchmark Threshold</label>
          <input
            type="range"
            min="0.80"
            max="0.99"
            step="0.01"
            value={settings.quality_thresholds.core_benchmark}
            onChange={(e) => updateSetting('quality_thresholds.core_benchmark', parseFloat(e.target.value))}
            className="ml-4 w-48"
          />
          <span className="ml-2">{(settings.quality_thresholds.core_benchmark * 100).toFixed(0)}%</span>
        </div>

        <div>
          <label className="font-medium">Overall Retention Threshold</label>
          <input
            type="range"
            min="0.70"
            max="0.95"
            step="0.01"
            value={settings.quality_thresholds.overall}
            onChange={(e) => updateSetting('quality_thresholds.overall', parseFloat(e.target.value))}
            className="ml-4 w-48"
          />
          <span className="ml-2">{(settings.quality_thresholds.overall * 100).toFixed(0)}%</span>
        </div>

        {/* Fallback strategy */}
        <div>
          <label className="font-medium">Fallback Strategy</label>
          <select
            value={settings.fallback_strategy}
            onChange={(e) => updateSetting('fallback_strategy', e.target.value)}
            className="ml-4 px-3 py-2 border rounded"
          >
            <option value="auto">Automatic (use previous stage)</option>
            <option value="manual">Manual (prompt for decision)</option>
          </select>
        </div>

        {/* Enable hypercompression */}
        <div className="flex items-center justify-between">
          <label className="font-medium">Enable Hypercompression Stage</label>
          <input
            type="checkbox"
            checked={settings.enable_hypercompression}
            onChange={(e) => updateSetting('enable_hypercompression', e.target.checked)}
          />
        </div>
      </div>

      {/* Save/Cancel buttons */}
      <div className="mt-6 flex gap-3 justify-end">
        <button
          onClick={onClose}
          className="px-4 py-2 border border-gray-400 rounded hover:bg-gray-100"
        >
          Cancel
        </button>
        <button
          onClick={() => { onSave(settings); onClose(); }}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Save Settings
        </button>
      </div>
    </Modal>
  );
};
```

---

## 3. WebSocket Integration

### 3.1 Real-Time Updates

**Backend WebSocket Events**:
```typescript
// Server-side event emitter
socket.emit('phase8:stage_start', {
  stage: 'vptq',
  timestamp: new Date(),
  estimated_duration_minutes: 180
});

socket.emit('phase8:compression_progress', {
  stage: 'vptq',
  progress: 0.67,  // 67%
  current_size_mb: 2.3,
  elapsed_seconds: 7200
});

socket.emit('phase8:benchmark_result', {
  stage: 'vptq',
  benchmark: 'mmlu',
  score: 0.735,
  retention: 0.971,
  passed: true
});

socket.emit('phase8:quality_gate', {
  stage: 'vptq',
  passed: false,
  failed_benchmarks: ['mmlu', 'gsm8k'],
  recommendations: {
    action: 'rollback_and_adjust',
    adjustments: { calibration_samples: 'increase by 20%' }
  }
});

socket.emit('phase8:retry_start', {
  stage: 'vptq',
  retry_number: 1,
  adjustments: { calibration_samples: 1200 }
});

socket.emit('phase8:stage_complete', {
  stage: 'vptq',
  passed: true,
  final_size_mb: 2.5,
  compression_ratio: 40,
  duration_seconds: 10800
});

socket.emit('phase8:phase_complete', {
  final_stage: 'hypercompression',
  final_size_mb: 0.4,
  total_compression_ratio: 250,
  total_duration_seconds: 97200,
  overall_passed: true
});
```

**Frontend Listener**:
```typescript
useEffect(() => {
  socket.on('phase8:stage_start', (data) => {
    setCurrentStage(data.stage);
    addEvent({
      timestamp: new Date(data.timestamp),
      level: 'info',
      stage: data.stage,
      message: `Starting ${data.stage} compression...`
    });
  });

  socket.on('phase8:compression_progress', (data) => {
    setProgress(data.progress * 100);
    setCurrentSize(data.current_size_mb);
    setElapsedTime(data.elapsed_seconds);
  });

  socket.on('phase8:benchmark_result', (data) => {
    updateBenchmarkResult(data.stage, data.benchmark, data.score);
    addEvent({
      timestamp: new Date(),
      level: data.passed ? 'success' : 'warning',
      stage: data.stage,
      message: `${data.benchmark.toUpperCase()}: ${data.score.toFixed(3)} (${(data.retention * 100).toFixed(1)}% retention)`
    });
  });

  socket.on('phase8:quality_gate', (data) => {
    if (!data.passed) {
      addEvent({
        timestamp: new Date(),
        level: 'warning',
        stage: data.stage,
        message: `Quality gate failed. Failed benchmarks: ${data.failed_benchmarks.join(', ')}`,
        details: data.recommendations
      });
    } else {
      addEvent({
        timestamp: new Date(),
        level: 'success',
        stage: data.stage,
        message: 'Quality gate passed ✅'
      });
    }
  });

  return () => {
    socket.off('phase8:stage_start');
    socket.off('phase8:compression_progress');
    socket.off('phase8:benchmark_result');
    socket.off('phase8:quality_gate');
  };
}, []);
```

---

## 4. REST API Endpoints

### 4.1 Phase 8 Endpoints

```typescript
// Start Phase 8 compression
POST /api/phase8/start
Body: {
  model_path: string;
  expert_config: object;
  settings: CompressionSettings;
}
Response: {
  job_id: string;
  estimated_duration_hours: number;
}

// Get current status
GET /api/phase8/status/:job_id
Response: {
  current_stage: string;
  progress: number;
  elapsed_seconds: number;
  current_size_mb: number;
  events: Event[];
}

// Get benchmark results
GET /api/phase8/benchmarks/:job_id
Response: {
  baseline: { mmlu: 0.757, gsm8k: 0.685, ... };
  seedlm: { mmlu: 0.748, gsm8k: 0.679, ... };
  vptq: { mmlu: 0.735, gsm8k: 0.661, ... };
  hypercompression: { mmlu: 0.722, gsm8k: 0.643, ... };
}

// Pause/resume
POST /api/phase8/pause/:job_id
POST /api/phase8/resume/:job_id

// Abort
POST /api/phase8/abort/:job_id
Body: { reason: string; }

// Export results
GET /api/phase8/export/:job_id
Response: CSV/JSON download with full benchmark comparison

// Get recommendations (if quality gate failed)
GET /api/phase8/recommendations/:job_id/:stage
Response: {
  action: string;
  adjustments: object;
  reason: string;
}
```

---

## 5. Mobile Responsive Design

### 5.1 Breakpoints

```css
/* Tailwind CSS breakpoints */
sm: 640px   /* Mobile landscape */
md: 768px   /* Tablet */
lg: 1024px  /* Desktop */
xl: 1280px  /* Large desktop */
```

### 5.2 Mobile Layout Adaptations

**Desktop (lg+)**: 3-column grid
**Tablet (md)**: 2-column grid
**Mobile (sm)**: Single-column stack

```typescript
// Responsive grid
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
  <CompressionProgressChart />
  <QualityMetrics />
  <BenchmarkResultsTable />
</div>
```

**Mobile Optimizations**:
- Collapsible sections (expand/collapse panels)
- Swipeable tabs for different views
- Simplified charts (smaller data points)
- Bottom-sheet controls instead of fixed footer
- Touch-friendly buttons (min 44×44px)

---

## 6. Accessibility (WCAG 2.1 AA)

### 6.1 Requirements

- ✅ **Keyboard Navigation**: All controls accessible via Tab/Enter
- ✅ **Screen Reader Support**: ARIA labels on all interactive elements
- ✅ **Color Contrast**: Minimum 4.5:1 for text, 3:1 for UI components
- ✅ **Focus Indicators**: Visible outline on focused elements
- ✅ **Alternative Text**: Descriptive labels for charts/graphs
- ✅ **Semantic HTML**: Proper heading hierarchy (h1 → h2 → h3)

### 6.2 Implementation Examples

```typescript
// Accessible button
<button
  aria-label="Start Phase 8 compression"
  aria-describedby="phase8-description"
  className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-4 focus:ring-blue-300"
>
  Start Phase 8
</button>

// Accessible gauge
<div
  role="progressbar"
  aria-valuenow={currentRetention}
  aria-valuemin={0}
  aria-valuemax={1}
  aria-label="Overall quality retention"
>
  <CircularGauge value={currentRetention} />
</div>

// Accessible table
<table role="table" aria-label="Benchmark comparison across compression stages">
  <thead>
    <tr>
      <th scope="col">Benchmark</th>
      <th scope="col">Baseline</th>
      ...
    </tr>
  </thead>
  <tbody>
    {/* rows */}
  </tbody>
</table>
```

---

## 7. W&B Dashboard Integration

### 7.1 Embedded W&B Panels

```typescript
// Embed W&B chart in UI
<div className="p-4 bg-white rounded-lg shadow">
  <h4 className="text-lg font-semibold mb-4">W&B Live Metrics</h4>
  <iframe
    src={`https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}/runs/${runId}?workspace=user-username`}
    width="100%"
    height="500px"
    frameBorder="0"
  />
</div>
```

### 7.2 W&B Links

Quick links to detailed W&B dashboards:

```typescript
<div className="flex gap-4 mt-4">
  <a
    href={`https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}/runs/${runId}`}
    target="_blank"
    className="text-blue-600 hover:underline"
  >
    📊 View Full W&B Dashboard
  </a>
  <a
    href={`https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}/runs/${runId}/logs`}
    target="_blank"
    className="text-blue-600 hover:underline"
  >
    📜 View W&B Logs
  </a>
  <a
    href={`https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}/artifacts`}
    target="_blank"
    className="text-blue-600 hover:underline"
  >
    📦 View Model Artifacts
  </a>
</div>
```

---

## 8. Implementation Timeline

### 8.1 Phase 8 UI Development Schedule

**Week 1**: Core components (1-2 days per component)
- Progress timeline
- Current stage card
- Compression progress chart
- Quality metrics dashboard

**Week 2**: Advanced components
- Benchmark results table
- Quality vs compression chart
- Expert performance radar
- Integration tests panel

**Week 3**: Infrastructure
- WebSocket integration
- REST API endpoints
- Event log system
- Control panel

**Week 4**: Polish & Testing
- Mobile responsive design
- Accessibility compliance
- W&B integration
- End-to-end testing

**Total**: 4 weeks for full Phase 8 UI implementation

---

## 9. Technology Stack

### 9.1 Frontend

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Charts**: Recharts (line, radar, gauge)
- **Real-time**: Socket.IO Client
- **State Management**: React Context + useReducer
- **HTTP Client**: Fetch API

### 9.2 Backend

- **Framework**: FastAPI (Python)
- **WebSocket**: python-socketio
- **W&B**: wandb Python SDK
- **Task Queue**: Celery (for long-running compression)
- **Database**: SQLite (job metadata)

### 9.3 Deployment

- **Frontend**: Vercel
- **Backend**: Docker container on cloud GPU instance
- **Reverse Proxy**: Nginx (WebSocket support)

---

## 10. Success Criteria

Phase 8 UI succeeds if:

1. ✅ **Real-time Updates**: WebSocket latency <100ms, updates every 2-5 seconds
2. ✅ **Quality Visualization**: All 7 core benchmarks + expert benchmarks visible
3. ✅ **Integration Tests**: Edge-of-chaos and eudaimonia displayed with visual indicators
4. ✅ **Event Tracking**: Comprehensive log with quality gate failures, retries, successes
5. ✅ **Mobile Support**: Fully responsive on 320px+ screens
6. ✅ **Accessibility**: WCAG 2.1 AA compliant (keyboard nav, screen readers, contrast)
7. ✅ **W&B Integration**: Links to detailed W&B dashboards, embedded panels
8. ✅ **User Control**: Start/pause/abort/export functionality
9. ✅ **Performance**: Dashboard loads in <2 seconds, handles 1000+ events without lag

---

## 11. Conclusion

This Phase 8 UI specification provides a comprehensive monitoring and visualization system for the 3-stage compression pipeline. Key features:

1. **Real-time Monitoring**: WebSocket-powered live updates of compression progress
2. **Quality Validation**: Visual display of benchmark testing results with threshold indicators
3. **Integration Testing**: Phase 5 edge-of-chaos and eudaimonia preserved and displayed
4. **Expert Performance**: Radar chart showing domain-specific benchmark retention
5. **Quality vs Compression Tradeoff**: Interactive chart showing accuracy loss at each stage
6. **Event Tracking**: Comprehensive log of quality gates, retries, failures, successes
7. **User Controls**: Start/pause/abort/export with configurable settings
8. **Accessibility**: WCAG 2.1 AA compliant for inclusive access
9. **W&B Integration**: Links and embeds for detailed experiment tracking

**User Requirement Met**: "has a ui component" for Phase 8 compression monitoring ✅

This UI enables users to monitor Phase 8 compression in real-time, ensuring quality preservation while achieving maximum compression (280×).
