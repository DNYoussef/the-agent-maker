# Agent Forge Frontend Architecture
## Component Architecture for 8-Phase AI Pipeline

**Version:** 1.0.0
**Stack:** Next.js 14 App Router, React 19, TypeScript 5, Tailwind CSS 4
**State Management:** Zustand + TanStack Query
**Real-time:** WebSocket + Server-Sent Events
**3D Visualization:** Three.js + React Three Fiber

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Directory Structure](#directory-structure)
3. [Type System](#type-system)
4. [State Management Strategy](#state-management-strategy)
5. [WebSocket Integration](#websocket-integration)
6. [Shared Components](#shared-components)
7. [Phase Control Patterns](#phase-control-patterns)
8. [Performance Optimization](#performance-optimization)
9. [Testing Strategy](#testing-strategy)

---

## 1. Architecture Overview

### Design Principles
- **Modular Component Design**: Reusable phase controls with consistent interfaces
- **Type Safety First**: Comprehensive TypeScript definitions for all data flows
- **Real-time by Default**: WebSocket connections for live pipeline updates
- **Responsive & Accessible**: Mobile-first design with WCAG 2.1 AA compliance
- **Performance Optimized**: Code splitting, lazy loading, and optimistic updates

### Current Stack Analysis
```typescript
// Existing Dependencies (package.json)
{
  "next": "15.5.4",                    // App Router with React Server Components
  "react": "19.1.0",                   // Latest with Concurrent Features
  "zustand": "^5.0.8",                 // Lightweight state management
  "@tanstack/react-query": "^5.90.2", // Server state & caching
  "lucide-react": "^0.544.0",          // Icon system
  "@react-three/fiber": "^9.3.0",      // 3D visualization
  "@react-three/drei": "^10.7.6",      // 3D helpers
  "framer-motion": "^12.23.19",        // Animations
  "recharts": "^3.2.1"                 // Charts (underutilized)
}
```

### Recommended State Strategy
**Zustand for Client State** + **TanStack Query for Server State**

**Why Not Redux?**
- Zustand is already installed and lighter (1.2kb vs 8kb)
- Less boilerplate for the 8-phase pipeline use case
- Better DevX with TypeScript inference
- React Query handles server caching/sync automatically

---

## 2. Directory Structure

### Proposed Organization
```
src/web/dashboard/
├── app/                          # Next.js App Router
│   ├── (dashboard)/              # Route group for authenticated layout
│   │   ├── layout.tsx            # Dashboard shell with navigation
│   │   ├── page.tsx              # Main dashboard (already exists)
│   │   └── phases/
│   │       ├── [phaseId]/        # Dynamic phase route
│   │       │   ├── page.tsx      # Phase detail page
│   │       │   └── loading.tsx   # Suspense fallback
│   │       └── layout.tsx        # Shared phase controls
│   ├── api/                      # API routes
│   │   ├── stats/route.ts        # Dashboard stats (exists)
│   │   ├── pipeline/
│   │   │   ├── [pipelineId]/route.ts
│   │   │   └── start/route.ts
│   │   └── ws/route.ts           # WebSocket upgrade
│   ├── layout.tsx                # Root layout (exists)
│   └── globals.css               # Global styles (exists)
│
├── components/                   # Shared components
│   ├── ui/                       # Primitive UI components
│   │   ├── Button.tsx
│   │   ├── Card.tsx
│   │   ├── Input.tsx
│   │   ├── Select.tsx
│   │   ├── Badge.tsx
│   │   └── Skeleton.tsx
│   ├── phase/                    # Phase-specific components
│   │   ├── PhaseCard.tsx
│   │   ├── PhaseControls.tsx
│   │   ├── PhaseHeader.tsx
│   │   ├── PhaseMetrics.tsx
│   │   ├── PhaseProgress.tsx
│   │   └── PhaseVisualizer.tsx
│   ├── pipeline/                 # Pipeline components
│   │   ├── PipelineRunner.tsx
│   │   ├── PipelineTimeline.tsx
│   │   └── PipelineStatus.tsx
│   ├── charts/                   # Data visualization
│   │   ├── MetricsChart.tsx
│   │   ├── ProgressRing.tsx
│   │   └── TimeSeriesChart.tsx
│   └── layout/                   # Layout components
│       ├── Header.tsx
│       ├── Sidebar.tsx
│       └── Footer.tsx
│
├── lib/                          # Utilities & configurations
│   ├── api/                      # API client
│   │   ├── client.ts             # Base fetch wrapper
│   │   ├── queries.ts            # TanStack Query hooks
│   │   └── mutations.ts          # TanStack Mutation hooks
│   ├── websocket/                # WebSocket management
│   │   ├── WebSocketProvider.tsx
│   │   ├── useWebSocket.ts
│   │   └── events.ts
│   ├── store/                    # Zustand stores
│   │   ├── pipelineStore.ts
│   │   ├── phaseStore.ts
│   │   └── uiStore.ts
│   ├── utils/                    # Helper functions
│   │   ├── formatters.ts
│   │   ├── validators.ts
│   │   └── constants.ts
│   └── hooks/                    # Custom React hooks
│       ├── usePhaseControl.ts
│       ├── usePipelineStatus.ts
│       └── useRealTimeMetrics.ts
│
├── types/                        # TypeScript definitions
│   ├── pipeline.ts
│   ├── phase.ts
│   ├── metrics.ts
│   ├── api.ts
│   └── websocket.ts
│
└── __tests__/                    # Test files
    ├── components/
    ├── integration/
    └── e2e/
```

---

## 3. Type System

### Core Type Definitions

```typescript
// types/pipeline.ts
export type PipelineStatus = 'idle' | 'running' | 'paused' | 'completed' | 'failed';

export type PhaseId =
  | 'cognate'      // Phase 1: Model Creation
  | 'evomerge'     // Phase 2: Evolution
  | 'quietstar'    // Phase 3: Reasoning
  | 'bitnet'       // Phase 4: Compression
  | 'forge'        // Phase 5: Training
  | 'baking'       // Phase 6: Tools
  | 'adas'         // Phase 7: Architecture
  | 'final';       // Phase 8: Production

export interface PhaseConfig {
  id: PhaseId;
  name: string;
  description: string;
  icon: string;
  color: string;
  order: number;
  dependencies: PhaseId[];
  estimatedDuration: number; // seconds
}

export interface PhaseState {
  id: PhaseId;
  status: PipelineStatus;
  progress: number; // 0-100
  startTime?: string;
  endTime?: string;
  metrics: Record<string, number | string>;
  logs: LogEntry[];
  error?: PipelineError;
}

export interface Pipeline {
  id: string;
  name: string;
  status: PipelineStatus;
  phases: PhaseState[];
  currentPhase?: PhaseId;
  createdAt: string;
  updatedAt: string;
  config: PipelineConfig;
}

export interface PipelineConfig {
  modelSelection: string[];
  trainingParams: TrainingParams;
  toolIntegrations: string[];
  targetPlatform: 'web' | 'cli' | 'api';
}

export interface TrainingParams {
  learningRate: number;
  batchSize: number;
  epochs: number;
  optimizer: 'adam' | 'sgd' | 'adamw';
}

export interface LogEntry {
  timestamp: string;
  level: 'info' | 'warn' | 'error' | 'debug';
  message: string;
  metadata?: Record<string, unknown>;
}

export interface PipelineError {
  code: string;
  message: string;
  phase: PhaseId;
  timestamp: string;
  stack?: string;
}
```

```typescript
// types/metrics.ts
export interface PhaseMetrics {
  phaseId: PhaseId;
  performance: PerformanceMetrics;
  quality: QualityMetrics;
  resources: ResourceMetrics;
}

export interface PerformanceMetrics {
  latency: number;
  throughput: number;
  successRate: number;
  errorRate: number;
}

export interface QualityMetrics {
  accuracy?: number;
  loss?: number;
  f1Score?: number;
  precision?: number;
  recall?: number;
}

export interface ResourceMetrics {
  cpuUsage: number;
  memoryUsage: number;
  gpuUsage?: number;
  diskIO: number;
}

export interface DashboardStats {
  totalAgents: number;
  successRate: number;
  activePipelines: number;
  avgPipelineTime: number;
  recentPipelines: Pipeline[];
}
```

```typescript
// types/websocket.ts
export type WebSocketEvent =
  | 'pipeline:started'
  | 'pipeline:completed'
  | 'pipeline:failed'
  | 'phase:started'
  | 'phase:progress'
  | 'phase:completed'
  | 'phase:failed'
  | 'metrics:update'
  | 'log:entry';

export interface WebSocketMessage<T = unknown> {
  event: WebSocketEvent;
  data: T;
  timestamp: string;
  pipelineId?: string;
}

export interface PhaseProgressEvent {
  phaseId: PhaseId;
  progress: number;
  metrics?: Partial<PhaseMetrics>;
  message?: string;
}

export interface MetricsUpdateEvent {
  phaseId: PhaseId;
  metrics: PhaseMetrics;
}
```

```typescript
// types/api.ts
export interface ApiResponse<T> {
  data: T;
  error?: ApiError;
  metadata?: {
    timestamp: string;
    requestId: string;
  };
}

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasMore: boolean;
}

// API Request Types
export interface CreatePipelineRequest {
  name: string;
  config: PipelineConfig;
}

export interface UpdatePhaseRequest {
  phaseId: PhaseId;
  config: Partial<PhaseConfig>;
}

export interface StartPipelineRequest {
  pipelineId: string;
  fromPhase?: PhaseId;
}
```

---

## 4. State Management Strategy

### Zustand Store Architecture

```typescript
// lib/store/pipelineStore.ts
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { Pipeline, PhaseState, PhaseId } from '@/types/pipeline';

interface PipelineState {
  // State
  activePipeline: Pipeline | null;
  pipelines: Pipeline[];
  isRunning: boolean;

  // Actions
  setActivePipeline: (pipeline: Pipeline | null) => void;
  updatePhaseState: (phaseId: PhaseId, state: Partial<PhaseState>) => void;
  startPipeline: (pipelineId: string) => void;
  pausePipeline: () => void;
  resumePipeline: () => void;
  stopPipeline: () => void;
  addPipeline: (pipeline: Pipeline) => void;
  removePipeline: (pipelineId: string) => void;
}

export const usePipelineStore = create<PipelineState>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        activePipeline: null,
        pipelines: [],
        isRunning: false,

        // Actions
        setActivePipeline: (pipeline) =>
          set({ activePipeline: pipeline }),

        updatePhaseState: (phaseId, phaseUpdate) =>
          set((state) => {
            if (!state.activePipeline) return state;

            const phases = state.activePipeline.phases.map(phase =>
              phase.id === phaseId ? { ...phase, ...phaseUpdate } : phase
            );

            return {
              activePipeline: {
                ...state.activePipeline,
                phases,
                updatedAt: new Date().toISOString()
              }
            };
          }),

        startPipeline: (pipelineId) => {
          const pipeline = get().pipelines.find(p => p.id === pipelineId);
          if (pipeline) {
            set({
              activePipeline: { ...pipeline, status: 'running' },
              isRunning: true
            });
          }
        },

        pausePipeline: () =>
          set((state) => ({
            activePipeline: state.activePipeline
              ? { ...state.activePipeline, status: 'paused' }
              : null,
            isRunning: false
          })),

        resumePipeline: () =>
          set((state) => ({
            activePipeline: state.activePipeline
              ? { ...state.activePipeline, status: 'running' }
              : null,
            isRunning: true
          })),

        stopPipeline: () =>
          set({ activePipeline: null, isRunning: false }),

        addPipeline: (pipeline) =>
          set((state) => ({ pipelines: [...state.pipelines, pipeline] })),

        removePipeline: (pipelineId) =>
          set((state) => ({
            pipelines: state.pipelines.filter(p => p.id !== pipelineId)
          }))
      }),
      { name: 'pipeline-storage' }
    )
  )
);
```

```typescript
// lib/store/uiStore.ts
import { create } from 'zustand';
import type { PhaseId } from '@/types/pipeline';

interface UIState {
  // Sidebar & Navigation
  sidebarOpen: boolean;
  activePhase: PhaseId | null;

  // Modals & Dialogs
  showConfigModal: boolean;
  showLogsModal: boolean;

  // Notifications
  notifications: Notification[];

  // Actions
  toggleSidebar: () => void;
  setActivePhase: (phaseId: PhaseId | null) => void;
  openConfigModal: () => void;
  closeConfigModal: () => void;
  addNotification: (notification: Omit<Notification, 'id'>) => void;
  removeNotification: (id: string) => void;
}

interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  message: string;
  duration?: number;
}

export const useUIStore = create<UIState>((set) => ({
  sidebarOpen: true,
  activePhase: null,
  showConfigModal: false,
  showLogsModal: false,
  notifications: [],

  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  setActivePhase: (phaseId) => set({ activePhase: phaseId }),
  openConfigModal: () => set({ showConfigModal: true }),
  closeConfigModal: () => set({ showConfigModal: false }),

  addNotification: (notification) =>
    set((state) => ({
      notifications: [
        ...state.notifications,
        { ...notification, id: crypto.randomUUID() }
      ]
    })),

  removeNotification: (id) =>
    set((state) => ({
      notifications: state.notifications.filter(n => n.id !== id)
    }))
}));
```

### TanStack Query Integration

```typescript
// lib/api/queries.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { Pipeline, DashboardStats } from '@/types/pipeline';
import { apiClient } from './client';

// Query Keys
export const queryKeys = {
  pipelines: ['pipelines'] as const,
  pipeline: (id: string) => ['pipelines', id] as const,
  stats: ['stats'] as const,
  metrics: (phaseId: string) => ['metrics', phaseId] as const
};

// Dashboard Stats
export function useDashboardStats() {
  return useQuery({
    queryKey: queryKeys.stats,
    queryFn: () => apiClient.get<DashboardStats>('/api/stats'),
    refetchInterval: 3000, // Refresh every 3s
    staleTime: 2000
  });
}

// Pipeline Queries
export function usePipeline(pipelineId: string) {
  return useQuery({
    queryKey: queryKeys.pipeline(pipelineId),
    queryFn: () => apiClient.get<Pipeline>(`/api/pipeline/${pipelineId}`),
    enabled: !!pipelineId
  });
}

export function usePipelines() {
  return useQuery({
    queryKey: queryKeys.pipelines,
    queryFn: () => apiClient.get<Pipeline[]>('/api/pipeline')
  });
}

// Pipeline Mutations
export function useStartPipeline() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (pipelineId: string) =>
      apiClient.post(`/api/pipeline/${pipelineId}/start`),
    onSuccess: (_, pipelineId) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.pipeline(pipelineId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.stats });
    }
  });
}

export function useCreatePipeline() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: CreatePipelineRequest) =>
      apiClient.post<Pipeline>('/api/pipeline', data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.pipelines });
    }
  });
}
```

```typescript
// lib/api/client.ts
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || '';

class ApiClient {
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      }
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || 'API request failed');
    }

    return response.json();
  }

  async get<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'GET' });
  }

  async post<T>(endpoint: string, data?: unknown): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }

  async put<T>(endpoint: string, data?: unknown): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: JSON.stringify(data)
    });
  }

  async delete<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'DELETE' });
  }
}

export const apiClient = new ApiClient();
```

---

## 5. WebSocket Integration

### WebSocket Provider Pattern

```typescript
// lib/websocket/WebSocketProvider.tsx
'use client';

import { createContext, useContext, useEffect, useRef, useState, ReactNode } from 'react';
import type { WebSocketMessage, WebSocketEvent } from '@/types/websocket';

type MessageHandler = (message: WebSocketMessage) => void;
type EventHandlers = Map<WebSocketEvent, Set<MessageHandler>>;

interface WebSocketContextValue {
  isConnected: boolean;
  subscribe: (event: WebSocketEvent, handler: MessageHandler) => () => void;
  send: (message: WebSocketMessage) => void;
  reconnect: () => void;
}

const WebSocketContext = createContext<WebSocketContextValue | null>(null);

export function WebSocketProvider({
  children,
  url = 'ws://localhost:3000/api/ws'
}: {
  children: ReactNode;
  url?: string;
}) {
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const handlersRef = useRef<EventHandlers>(new Map());
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();

  const connect = () => {
    const ws = new WebSocket(url);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        const handlers = handlersRef.current.get(message.event);

        if (handlers) {
          handlers.forEach(handler => handler(message));
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);

      // Attempt reconnect after 3s
      reconnectTimeoutRef.current = setTimeout(() => {
        console.log('Attempting to reconnect...');
        connect();
      }, 3000);
    };

    wsRef.current = ws;
  };

  const subscribe = (event: WebSocketEvent, handler: MessageHandler) => {
    if (!handlersRef.current.has(event)) {
      handlersRef.current.set(event, new Set());
    }

    handlersRef.current.get(event)!.add(handler);

    // Return unsubscribe function
    return () => {
      const handlers = handlersRef.current.get(event);
      if (handlers) {
        handlers.delete(handler);
        if (handlers.size === 0) {
          handlersRef.current.delete(event);
        }
      }
    };
  };

  const send = (message: WebSocketMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected, message not sent:', message);
    }
  };

  const reconnect = () => {
    wsRef.current?.close();
    connect();
  };

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      wsRef.current?.close();
    };
  }, [url]);

  return (
    <WebSocketContext.Provider value={{ isConnected, subscribe, send, reconnect }}>
      {children}
    </WebSocketContext.Provider>
  );
}

export function useWebSocket() {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within WebSocketProvider');
  }
  return context;
}
```

### WebSocket Custom Hooks

```typescript
// lib/hooks/usePipelineStatus.ts
import { useEffect } from 'react';
import { useWebSocket } from '../websocket/WebSocketProvider';
import { usePipelineStore } from '../store/pipelineStore';
import type { PhaseProgressEvent, Pipeline } from '@/types';

export function usePipelineStatus(pipelineId?: string) {
  const { subscribe } = useWebSocket();
  const { updatePhaseState, setActivePipeline } = usePipelineStore();

  useEffect(() => {
    if (!pipelineId) return;

    const unsubscribeProgress = subscribe('phase:progress', (message) => {
      const data = message.data as PhaseProgressEvent;
      if (message.pipelineId === pipelineId) {
        updatePhaseState(data.phaseId, {
          progress: data.progress,
          metrics: data.metrics || {}
        });
      }
    });

    const unsubscribeCompleted = subscribe('pipeline:completed', (message) => {
      if (message.pipelineId === pipelineId) {
        const pipeline = message.data as Pipeline;
        setActivePipeline({ ...pipeline, status: 'completed' });
      }
    });

    const unsubscribeFailed = subscribe('pipeline:failed', (message) => {
      if (message.pipelineId === pipelineId) {
        const pipeline = message.data as Pipeline;
        setActivePipeline({ ...pipeline, status: 'failed' });
      }
    });

    return () => {
      unsubscribeProgress();
      unsubscribeCompleted();
      unsubscribeFailed();
    };
  }, [pipelineId, subscribe, updatePhaseState, setActivePipeline]);
}
```

```typescript
// lib/hooks/useRealTimeMetrics.ts
import { useState, useEffect } from 'react';
import { useWebSocket } from '../websocket/WebSocketProvider';
import type { PhaseMetrics, PhaseId, MetricsUpdateEvent } from '@/types';

export function useRealTimeMetrics(phaseId: PhaseId) {
  const [metrics, setMetrics] = useState<PhaseMetrics | null>(null);
  const { subscribe } = useWebSocket();

  useEffect(() => {
    const unsubscribe = subscribe('metrics:update', (message) => {
      const data = message.data as MetricsUpdateEvent;
      if (data.phaseId === phaseId) {
        setMetrics(data.metrics);
      }
    });

    return unsubscribe;
  }, [phaseId, subscribe]);

  return metrics;
}
```

---

## 6. Shared Components

### Phase Control Components

```typescript
// components/phase/PhaseControls.tsx
'use client';

import { Play, Pause, Square, RotateCw } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { usePipelineStore } from '@/lib/store/pipelineStore';
import type { PhaseId } from '@/types/pipeline';

interface PhaseControlsProps {
  phaseId: PhaseId;
  disabled?: boolean;
}

export function PhaseControls({ phaseId, disabled }: PhaseControlsProps) {
  const {
    activePipeline,
    isRunning,
    startPipeline,
    pausePipeline,
    resumePipeline,
    stopPipeline
  } = usePipelineStore();

  const phase = activePipeline?.phases.find(p => p.id === phaseId);
  const isActive = activePipeline?.currentPhase === phaseId;
  const canStart = !isRunning && phase?.status === 'idle';
  const canPause = isRunning && isActive;
  const canResume = !isRunning && phase?.status === 'paused';

  return (
    <div className="flex items-center gap-2">
      {canStart && (
        <Button
          onClick={() => activePipeline && startPipeline(activePipeline.id)}
          disabled={disabled}
          variant="primary"
          size="sm"
        >
          <Play className="w-4 h-4 mr-2" />
          Start
        </Button>
      )}

      {canPause && (
        <Button
          onClick={pausePipeline}
          disabled={disabled}
          variant="secondary"
          size="sm"
        >
          <Pause className="w-4 h-4 mr-2" />
          Pause
        </Button>
      )}

      {canResume && (
        <Button
          onClick={resumePipeline}
          disabled={disabled}
          variant="primary"
          size="sm"
        >
          <RotateCw className="w-4 h-4 mr-2" />
          Resume
        </Button>
      )}

      {isRunning && (
        <Button
          onClick={stopPipeline}
          disabled={disabled}
          variant="danger"
          size="sm"
        >
          <Square className="w-4 h-4 mr-2" />
          Stop
        </Button>
      )}
    </div>
  );
}
```

```typescript
// components/phase/PhaseProgress.tsx
'use client';

import { motion } from 'framer-motion';
import { CheckCircle, XCircle, Clock } from 'lucide-react';
import type { PhaseState } from '@/types/pipeline';

interface PhaseProgressProps {
  phase: PhaseState;
  showDetails?: boolean;
}

export function PhaseProgress({ phase, showDetails = true }: PhaseProgressProps) {
  const statusIcon = {
    idle: <Clock className="w-5 h-5 text-gray-400" />,
    running: <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 2, ease: "linear" }}>
      <Clock className="w-5 h-5 text-blue-400" />
    </motion.div>,
    paused: <Clock className="w-5 h-5 text-yellow-400" />,
    completed: <CheckCircle className="w-5 h-5 text-green-400" />,
    failed: <XCircle className="w-5 h-5 text-red-400" />
  };

  const statusColor = {
    idle: 'bg-gray-600',
    running: 'bg-blue-500',
    paused: 'bg-yellow-500',
    completed: 'bg-green-500',
    failed: 'bg-red-500'
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {statusIcon[phase.status]}
          <span className="text-sm font-medium capitalize">{phase.status}</span>
        </div>
        <span className="text-sm text-gray-400">{phase.progress}%</span>
      </div>

      <div className="relative h-2 bg-white/10 rounded-full overflow-hidden">
        <motion.div
          className={`absolute inset-y-0 left-0 ${statusColor[phase.status]}`}
          initial={{ width: 0 }}
          animate={{ width: `${phase.progress}%` }}
          transition={{ duration: 0.5 }}
        />
      </div>

      {showDetails && phase.startTime && (
        <div className="text-xs text-gray-500">
          Started: {new Date(phase.startTime).toLocaleString()}
        </div>
      )}
    </div>
  );
}
```

```typescript
// components/phase/PhaseMetrics.tsx
'use client';

import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import type { PhaseMetrics } from '@/types/metrics';

interface PhaseMetricsProps {
  metrics: PhaseMetrics;
  compact?: boolean;
}

export function PhaseMetrics({ metrics, compact = false }: PhaseMetricsProps) {
  const formatMetric = (value: number, suffix: string = '') => {
    return `${value.toFixed(2)}${suffix}`;
  };

  const getTrendIcon = (value: number) => {
    if (value > 0) return <TrendingUp className="w-4 h-4 text-green-400" />;
    if (value < 0) return <TrendingDown className="w-4 h-4 text-red-400" />;
    return <Minus className="w-4 h-4 text-gray-400" />;
  };

  if (compact) {
    return (
      <div className="grid grid-cols-2 gap-2">
        <MetricItem
          label="Latency"
          value={formatMetric(metrics.performance.latency, 'ms')}
        />
        <MetricItem
          label="Success"
          value={formatMetric(metrics.performance.successRate, '%')}
        />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div>
        <h4 className="text-sm font-medium text-gray-400 mb-2">Performance</h4>
        <div className="grid grid-cols-2 gap-3">
          <MetricItem
            label="Latency"
            value={formatMetric(metrics.performance.latency, 'ms')}
          />
          <MetricItem
            label="Throughput"
            value={formatMetric(metrics.performance.throughput, '/s')}
          />
          <MetricItem
            label="Success Rate"
            value={formatMetric(metrics.performance.successRate, '%')}
          />
          <MetricItem
            label="Error Rate"
            value={formatMetric(metrics.performance.errorRate, '%')}
          />
        </div>
      </div>

      {metrics.quality.accuracy && (
        <div>
          <h4 className="text-sm font-medium text-gray-400 mb-2">Quality</h4>
          <div className="grid grid-cols-2 gap-3">
            {metrics.quality.accuracy && (
              <MetricItem
                label="Accuracy"
                value={formatMetric(metrics.quality.accuracy, '%')}
              />
            )}
            {metrics.quality.loss && (
              <MetricItem
                label="Loss"
                value={formatMetric(metrics.quality.loss)}
              />
            )}
          </div>
        </div>
      )}

      <div>
        <h4 className="text-sm font-medium text-gray-400 mb-2">Resources</h4>
        <div className="grid grid-cols-2 gap-3">
          <MetricItem
            label="CPU"
            value={formatMetric(metrics.resources.cpuUsage, '%')}
          />
          <MetricItem
            label="Memory"
            value={formatMetric(metrics.resources.memoryUsage, 'MB')}
          />
        </div>
      </div>
    </div>
  );
}

function MetricItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-white/5 rounded-lg p-2">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className="text-sm font-semibold">{value}</div>
    </div>
  );
}
```

### UI Primitives

```typescript
// components/ui/Button.tsx
import { ButtonHTMLAttributes, forwardRef } from 'react';
import { cva, type VariantProps } from 'class-variance-authority';

const buttonVariants = cva(
  'inline-flex items-center justify-center rounded-lg font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none',
  {
    variants: {
      variant: {
        primary: 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500',
        secondary: 'bg-gray-600 text-white hover:bg-gray-700 focus:ring-gray-500',
        danger: 'bg-red-600 text-white hover:bg-red-700 focus:ring-red-500',
        ghost: 'bg-transparent hover:bg-white/10 text-gray-300'
      },
      size: {
        sm: 'h-8 px-3 text-sm',
        md: 'h-10 px-4',
        lg: 'h-12 px-6 text-lg'
      }
    },
    defaultVariants: {
      variant: 'primary',
      size: 'md'
    }
  }
);

export interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={buttonVariants({ variant, size, className })}
        {...props}
      />
    );
  }
);

Button.displayName = 'Button';
```

---

## 7. Phase Control Patterns

### Unified Phase Layout

```typescript
// app/(dashboard)/phases/layout.tsx
'use client';

import { ReactNode } from 'react';
import { PhaseControls } from '@/components/phase/PhaseControls';
import { usePipelineStore } from '@/lib/store/pipelineStore';

export default function PhaseLayout({
  children,
  params
}: {
  children: ReactNode;
  params: { phaseId: string };
}) {
  const { activePipeline } = usePipelineStore();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950">
      <div className="sticky top-0 z-10 bg-black/20 backdrop-blur-lg border-b border-white/10">
        <div className="container mx-auto px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link href="/" className="text-gray-400 hover:text-white">
                <ArrowLeft className="w-5 h-5" />
              </Link>
              <h1 className="text-2xl font-bold">
                {params.phaseId.charAt(0).toUpperCase() + params.phaseId.slice(1)}
              </h1>
            </div>
            {activePipeline && (
              <PhaseControls phaseId={params.phaseId as PhaseId} />
            )}
          </div>
        </div>
      </div>
      <div className="container mx-auto px-8 py-8">
        {children}
      </div>
    </div>
  );
}
```

### Dynamic Phase Page Template

```typescript
// app/(dashboard)/phases/[phaseId]/page.tsx
'use client';

import { use } from 'react';
import { PhaseProgress } from '@/components/phase/PhaseProgress';
import { PhaseMetrics } from '@/components/phase/PhaseMetrics';
import { usePipelineStore } from '@/lib/store/pipelineStore';
import { useRealTimeMetrics } from '@/lib/hooks/useRealTimeMetrics';
import type { PhaseId } from '@/types/pipeline';

export default function PhasePage({
  params
}: {
  params: Promise<{ phaseId: string }>
}) {
  const { phaseId } = use(params);
  const { activePipeline } = usePipelineStore();
  const metrics = useRealTimeMetrics(phaseId as PhaseId);

  const phase = activePipeline?.phases.find(p => p.id === phaseId);

  if (!phase) {
    return <div>Phase not found</div>;
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
      <div className="space-y-6">
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
          <h2 className="text-xl font-bold mb-4">Progress</h2>
          <PhaseProgress phase={phase} />
        </div>

        {metrics && (
          <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
            <h2 className="text-xl font-bold mb-4">Metrics</h2>
            <PhaseMetrics metrics={metrics} />
          </div>
        )}
      </div>

      <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
        <h2 className="text-xl font-bold mb-4">Logs</h2>
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {phase.logs.map((log, i) => (
            <div key={i} className="text-sm">
              <span className="text-gray-500">{new Date(log.timestamp).toLocaleTimeString()}</span>
              {' '}
              <span className={`font-medium ${
                log.level === 'error' ? 'text-red-400' :
                log.level === 'warn' ? 'text-yellow-400' :
                'text-gray-300'
              }`}>
                {log.message}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
```

---

## 8. Performance Optimization

### Code Splitting Strategy

```typescript
// app/(dashboard)/layout.tsx
import dynamic from 'next/dynamic';
import { Suspense } from 'react';

const WebSocketProvider = dynamic(() =>
  import('@/lib/websocket/WebSocketProvider').then(mod => mod.WebSocketProvider),
  { ssr: false }
);

const Sidebar = dynamic(() => import('@/components/layout/Sidebar'));

export default function DashboardLayout({ children }: { children: ReactNode }) {
  return (
    <WebSocketProvider>
      <div className="flex">
        <Suspense fallback={<SidebarSkeleton />}>
          <Sidebar />
        </Suspense>
        <main className="flex-1">
          {children}
        </main>
      </div>
    </WebSocketProvider>
  );
}
```

### Optimistic Updates

```typescript
// lib/api/mutations.ts
export function useUpdatePhase() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ phaseId, config }: UpdatePhaseRequest) =>
      apiClient.put(`/api/phase/${phaseId}`, config),

    onMutate: async (variables) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({ queryKey: queryKeys.pipeline(variables.phaseId) });

      // Snapshot previous value
      const previous = queryClient.getQueryData(queryKeys.pipeline(variables.phaseId));

      // Optimistically update
      queryClient.setQueryData(queryKeys.pipeline(variables.phaseId), (old: Pipeline) => ({
        ...old,
        phases: old.phases.map(p =>
          p.id === variables.phaseId ? { ...p, ...variables.config } : p
        )
      }));

      return { previous };
    },

    onError: (err, variables, context) => {
      // Rollback on error
      if (context?.previous) {
        queryClient.setQueryData(
          queryKeys.pipeline(variables.phaseId),
          context.previous
        );
      }
    },

    onSettled: (_, __, variables) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.pipeline(variables.phaseId) });
    }
  });
}
```

---

## 9. Testing Strategy

### Component Testing

```typescript
// __tests__/components/PhaseControls.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { PhaseControls } from '@/components/phase/PhaseControls';
import { usePipelineStore } from '@/lib/store/pipelineStore';

jest.mock('@/lib/store/pipelineStore');

describe('PhaseControls', () => {
  it('should render start button when phase is idle', () => {
    (usePipelineStore as jest.Mock).mockReturnValue({
      activePipeline: {
        id: 'test',
        currentPhase: 'cognate',
        phases: [{ id: 'cognate', status: 'idle' }]
      },
      isRunning: false,
      startPipeline: jest.fn()
    });

    render(<PhaseControls phaseId="cognate" />);

    expect(screen.getByText('Start')).toBeInTheDocument();
  });

  it('should call startPipeline when start button is clicked', () => {
    const startPipeline = jest.fn();
    (usePipelineStore as jest.Mock).mockReturnValue({
      activePipeline: { id: 'test', phases: [{ id: 'cognate', status: 'idle' }] },
      isRunning: false,
      startPipeline
    });

    render(<PhaseControls phaseId="cognate" />);

    fireEvent.click(screen.getByText('Start'));
    expect(startPipeline).toHaveBeenCalledWith('test');
  });
});
```

### Integration Testing

```typescript
// __tests__/integration/pipeline-flow.test.tsx
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { WebSocketProvider } from '@/lib/websocket/WebSocketProvider';
import PipelinePage from '@/app/(dashboard)/pipeline/[id]/page';

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: false } }
});

function Wrapper({ children }: { children: ReactNode }) {
  return (
    <QueryClientProvider client={queryClient}>
      <WebSocketProvider url="ws://localhost:3000/test">
        {children}
      </WebSocketProvider>
    </QueryClientProvider>
  );
}

describe('Pipeline Flow', () => {
  it('should update phase progress via WebSocket', async () => {
    const { container } = render(
      <PipelinePage params={{ id: 'test-pipeline' }} />,
      { wrapper: Wrapper }
    );

    // Simulate WebSocket message
    const wsMessage = {
      event: 'phase:progress',
      pipelineId: 'test-pipeline',
      data: { phaseId: 'cognate', progress: 50 }
    };

    // Trigger WebSocket event
    window.dispatchEvent(new MessageEvent('message', {
      data: JSON.stringify(wsMessage)
    }));

    await waitFor(() => {
      expect(screen.getByText('50%')).toBeInTheDocument();
    });
  });
});
```

---

## Next Steps

### Implementation Checklist

1. **Phase 1: Foundation**
   - [ ] Set up type definitions in `/types`
   - [ ] Create Zustand stores in `/lib/store`
   - [ ] Implement WebSocket provider
   - [ ] Build UI primitive components

2. **Phase 2: Core Components**
   - [ ] Build phase control components
   - [ ] Create metrics visualization components
   - [ ] Implement progress tracking UI
   - [ ] Add logging viewer

3. **Phase 3: Integration**
   - [ ] Integrate TanStack Query
   - [ ] Connect WebSocket events to stores
   - [ ] Implement optimistic updates
   - [ ] Add error boundaries

4. **Phase 4: Polish**
   - [ ] Add loading states and skeletons
   - [ ] Implement animations with Framer Motion
   - [ ] Optimize bundle size
   - [ ] Write comprehensive tests

5. **Phase 5: Production**
   - [ ] Performance profiling
   - [ ] Accessibility audit
   - [ ] Security review
   - [ ] Documentation

---

## File Paths Reference

```
Key Implementation Files:
├── /types/pipeline.ts              # Core type definitions
├── /types/metrics.ts               # Metrics types
├── /types/websocket.ts             # WebSocket event types
├── /lib/store/pipelineStore.ts     # Pipeline state management
├── /lib/store/uiStore.ts           # UI state management
├── /lib/websocket/WebSocketProvider.tsx  # WebSocket context
├── /lib/api/client.ts              # API client
├── /lib/api/queries.ts             # TanStack Query hooks
├── /components/phase/PhaseControls.tsx   # Phase control UI
├── /components/phase/PhaseProgress.tsx   # Progress visualization
├── /components/phase/PhaseMetrics.tsx    # Metrics display
└── /app/(dashboard)/phases/[phaseId]/page.tsx  # Dynamic phase page
```