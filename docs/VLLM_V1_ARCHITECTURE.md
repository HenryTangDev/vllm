# vLLM V1 Architecture Analysis

## Overview

vLLM V1 represents a significant architectural refactoring of the vLLM inference engine, designed for improved performance, scalability, and maintainability. This document provides a comprehensive analysis of the V1 architecture.

## High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Client Request                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API Layer (Entrypoints)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ OpenAI API  │  │   gRPC API  │  │   REST API  │  │  LLM (Offline)      │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Engine Layer (v1/engine/)                            │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    AsyncLLM / LLMEngine                                │  │
│  │  ┌─────────────┐  ┌───────────────┐  ┌────────────────────────────┐   │  │
│  │  │  Processor  │  │OutputProcessor│  │     EngineCoreClient       │   │  │
│  │  │ (Tokenize,  │  │ (Detokenize,  │  │  (Communication w/ Core)   │   │  │
│  │  │  Preprocess)│  │  Streaming)   │  │                            │   │  │
│  │  └─────────────┘  └───────────────┘  └────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                            ZMQ/IPC Communication
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EngineCore (v1/engine/core.py)                          │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                          Core Busy Loop                                │  │
│  │  1. Process Input Queue (add/abort requests)                          │  │
│  │  2. Schedule Requests                                                  │  │
│  │  3. Execute Model                                                      │  │
│  │  4. Update State & Return Outputs                                      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│          ┌───────────────────────────┼───────────────────────────┐          │
│          ▼                           ▼                           ▼          │
│  ┌───────────────┐         ┌─────────────────┐         ┌────────────────┐   │
│  │   Scheduler   │         │    Executor     │         │ Structured     │   │
│  │ (v1/core/sched)│        │ (v1/executor/)  │         │ Output Manager │   │
│  └───────────────┘         └─────────────────┘         └────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Worker Layer (v1/worker/)                            │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                          GPU Worker                                    │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     GPUModelRunner                               │  │  │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────────┐  │  │  │
│  │  │  │  Model   │  │InputBatch│  │ Sampler  │  │ Spec Decode     │  │  │  │
│  │  │  │ Execution│  │ Mgmt     │  │          │  │ (Eagle/Medusa)  │  │  │  │
│  │  │  └──────────┘  └──────────┘  └──────────┘  └─────────────────┘  │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Model & KV Cache Layer                                │
│  ┌───────────────────────┐         ┌────────────────────────────────────┐   │
│  │   Transformer Model   │         │        KV Cache Management         │   │
│  │   (model_executor/)   │◄───────►│  ┌─────────────────────────────┐   │   │
│  │                       │         │  │   KVCacheManager            │   │   │
│  │   Attention Layers    │         │  │   - Block Pool              │   │   │
│  │   MLP Layers          │         │  │   - Prefix Caching          │   │   │
│  │   Embedding Layers    │         │  │   - Block Allocation        │   │   │
│  └───────────────────────┘         │  └─────────────────────────────┘   │   │
│                                    └────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Engine Layer (`v1/engine/`)

#### AsyncLLM (`async_llm.py`)
The primary async interface for serving requests.

**Key Responsibilities:**
- Async request handling with `generate()` and `encode()` methods
- Background output handler task
- Request queue management
- LoRA adapter management
- Pause/resume generation for model updates

**Key Methods:**
```python
async def generate(prompt, sampling_params, request_id) -> AsyncGenerator[RequestOutput]
async def encode(prompt, pooling_params, request_id) -> AsyncGenerator[PoolingRequestOutput]
async def add_request(request_id, prompt, params) -> RequestOutputCollector
```

#### LLMEngine (`llm_engine.py`)
Synchronous wrapper for backward compatibility.

**Key Features:**
- Step-based execution model
- Integration with `EngineCoreClient`
- Output processing through `OutputProcessor`
- Data parallel support

#### EngineCore (`core.py`)
The heart of the inference engine running in a separate process.

**Key Responsibilities:**
- Main busy loop: schedule → execute → update
- Request preprocessing
- KV cache initialization
- Batch queue management for pipeline parallelism

**Core Loop:**
```python
def run_busy_loop(self):
    while True:
        self._process_input_queue()      # Handle new/abort requests
        self._process_engine_step()      # Schedule & execute
```

#### EngineCoreProc (`core.py`)
ZMQ-wrapped version for background process execution.

**Features:**
- Input/Output socket threads for async I/O
- Separate input processing thread
- Message serialization with msgspec

#### DPEngineCoreProc
Data parallel variant with synchronization support.

**Features:**
- Wave-based request processing
- All-reduce synchronization across DP ranks
- Request count publishing for load balancing

### 2. Scheduler (`v1/core/sched/`)

#### SchedulerInterface (`interface.py`)
Abstract interface defining scheduler contract.

**Key Methods:**
```python
def schedule() -> SchedulerOutput
def add_request(request: Request) -> None
def finish_requests(request_ids, finished_status) -> None
def update_from_output(scheduler_output, model_runner_output) -> dict[int, EngineCoreOutputs]
def get_grammar_bitmask(scheduler_output) -> GrammarOutput | None
```

#### Scheduler (`scheduler.py`)
Main scheduler implementation.

**Scheduling Algorithm:**
1. **No distinct phases** - Each request has `num_computed_tokens` and `num_tokens_with_spec`
2. **Iteration-level scheduling** - Determines tokens per request per step
3. **Supports**: Chunked prefill, prefix caching, speculative decoding

**Key Data Structures:**
- `waiting` queue (Priority queue by policy)
- `running` list
- `finished_req_ids` set

**Scheduling Constraints:**
- `max_num_running_reqs`
- `max_num_scheduled_tokens`
- `max_model_len`
- Encoder budget (for multimodal)

#### SchedulerOutput (`output.py`)
Encapsulates scheduling decisions.

**Contents:**
- `scheduled_new_reqs`: New requests to process
- `scheduled_cached_reqs`: Running/resumed requests
- `num_scheduled_tokens`: Token count per request
- `scheduled_spec_decode_tokens`: Speculative tokens
- `scheduled_encoder_inputs`: Multimodal inputs
- `kv_connector_metadata`: KV transfer metadata

### 3. Executor Layer (`v1/executor/`)

#### Executor Base (`abstract.py`)
Abstract base class for all executors.

**Executor Types:**
- `UniProcExecutor`: Single-process execution
- `MultiprocExecutor`: Multi-process for TP/PP
- `RayDistributedExecutor`: Ray-based distributed
- `ExecutorWithExternalLauncher`: External process launcher

**Key Methods:**
```python
def execute_model(scheduler_output) -> ModelRunnerOutput
def collective_rpc(method, args, kwargs) -> list[results]
def determine_available_memory() -> list[int]
```

#### MultiprocExecutor (`multiproc_executor.py`)
Multi-process executor with message queue communication.

**Features:**
- Worker process spawning
- MessageQueue for SchedulerOutput broadcast
- Response aggregation from workers
- Health monitoring thread

### 4. Worker Layer (`v1/worker/`)

#### Worker (`gpu_worker.py`)
GPU worker process implementation.

**Key Responsibilities:**
- Device initialization
- Model loading
- KV cache allocation
- Execute model steps
- Memory profiling

**Key Methods:**
```python
def execute_model(scheduler_output) -> ModelRunnerOutput
def determine_available_memory() -> int
def initialize_from_config(kv_cache_config) -> None
def compile_or_warm_up_model() -> None
```

#### GPUModelRunner (`gpu_model_runner.py`)
Model execution on GPU.

**Key Features:**
- Input batch preparation
- Attention metadata building
- CUDA graph capture
- Sampling & logprob computation
- Speculative decoding support
- LoRA adapter handling

**Execution Flow:**
```python
def execute_model(scheduler_output, intermediate_tensors):
    # 1. Prepare inputs
    # 2. Build attention metadata
    # 3. Run model forward
    # 4. Sample tokens (if enabled)
    # 5. Return outputs
```

### 5. KV Cache Management (`v1/core/`)

#### KVCacheManager (`kv_cache_manager.py`)
High-level KV cache management.

**Features:**
- Block allocation/deallocation
- Prefix caching support
- Block sharing across requests
- Usage tracking

**Key Methods:**
```python
def allocate_slots(request, num_tokens, num_lookahead_tokens) -> KVCacheBlocks
def free(request) -> None
def get_computed_blocks(request) -> tuple[KVCacheBlocks, int]
def cache_blocks(request, num_computed_tokens) -> None
```

#### BlockPool (`block_pool.py`)
Low-level block management.

**Features:**
- Free block tracking
- Block reference counting
- Hash-based prefix caching
- Event generation for KV transfers

### 6. Request Model (`v1/request.py`)

#### Request Class
Represents a single inference request.

**Key Attributes:**
```python
request_id: str
prompt_token_ids: list[int]
sampling_params: SamplingParams
num_computed_tokens: int
num_cached_tokens: int
status: RequestStatus
mm_features: list[MultiModalFeatureSpec]
```

**Request States:**
- `WAITING`: In scheduler queue
- `WAITING_FOR_FSM`: Structured output compilation
- `WAITING_FOR_REMOTE_KVS`: KV transfer in progress
- `RUNNING`: Currently executing
- `PREEMPTED`: Preempted, awaiting resume
- `FINISHED_*`: Various completion states

### 7. Outputs (`v1/outputs.py`)

#### ModelRunnerOutput
Results from model execution.

**Contents:**
- `sampled_token_ids`: Generated tokens
- `logprobs`: Token log probabilities
- `prompt_logprobs_dict`: Prompt token logprobs
- `pooler_output`: Embedding outputs
- `kv_connector_output`: KV transfer results

#### EngineCoreOutput
Per-request output sent to client.

**Contents:**
- `new_token_ids`: Newly generated tokens
- `finish_reason`: Completion reason
- `new_logprobs`: Token probabilities
- `kv_transfer_params`: KV connector metadata

## Data Flow

### Request Processing Flow

```
1. Client Request
   │
   ▼
2. Processor.process_inputs()
   - Tokenization
   - Multimodal preprocessing
   - Parameter validation
   │
   ▼
3. EngineCoreRequest created
   │
   ▼
4. Engine.add_request()
   │
   ▼
5. EngineCore.add_request()
   - Create Request object
   - Add to waiting queue
   │
   ▼
6. Scheduler.schedule()
   - Select requests to run
   - Allocate KV cache blocks
   - Build SchedulerOutput
   │
   ▼
7. Executor.execute_model()
   - Send to workers
   │
   ▼
8. Worker.execute_model()
   - Build input batch
   - Run model forward
   - Sample tokens
   │
   ▼
9. Scheduler.update_from_output()
   - Update request states
   - Check stop conditions
   - Build EngineCoreOutputs
   │
   ▼
10. OutputProcessor.process_outputs()
    - Detokenization
    - Streaming
    - Build RequestOutput
    │
    ▼
11. Return to Client
```

## Key Design Patterns

### 1. Process Separation
- **EngineCore**: Runs in separate process
- **Workers**: Each in own process
- **Communication**: ZMQ sockets with msgspec serialization

### 2. Async I/O
- Background threads for socket I/O
- Non-blocking operations
- Queue-based request batching

### 3. Batch Queue (Pipeline Parallelism)
```python
# Schedule while previous batch executes
scheduler_output = self.scheduler.schedule()
future = self.model_executor.execute_model(scheduler_output, non_block=True)
batch_queue.appendleft((future, scheduler_output))
```

### 4. Data Parallel Coordination
- Wave-based synchronization
- Coordinator for load balancing
- All-reduce for finish detection

### 5. Prefix Caching
- Hash-based block identification
- Block sharing across requests
- Lazy caching for computed blocks

## Speculative Decoding Support

### Supported Methods
- **EAGLE**: Tree-structured speculation
- **Medusa**: Multi-head prediction
- **N-gram**: Statistical prediction
- **Suffix Decoding**: Pattern-based speculation

### Integration Points
- `SchedulerOutput.scheduled_spec_decode_tokens`
- `Scheduler.update_draft_token_ids()`
- `GPUModelRunner.take_draft_token_ids()`

## Multimodal Support

### Components
- `MultiModalFeatureSpec`: Feature specification
- `EncoderCacheManager`: Encoder output caching
- `ECConnector`: External encoder caching

### Flow
1. Extract multimodal inputs from request
2. Schedule encoder inputs within budget
3. Process through vision encoder
4. Cache encoder outputs
5. Use in decoder attention

## KV Connector (Disaggregated Inference)

### Purpose
Support prefill-decode disaggregation and KV cache offloading.

### Components
- `KVConnectorBase_V1`: Base interface
- `KVConnectorMetadata`: Transfer metadata
- `KVOutputAggregator`: Output consolidation

### Request States for KV Transfer
- `WAITING_FOR_REMOTE_KVS`: Async KV loading
- Managed by scheduler with `finished_recving_kv_req_ids`

## Configuration Integration

### VllmConfig Components
- `model_config`: Model architecture settings
- `cache_config`: KV cache parameters
- `parallel_config`: TP/PP/DP settings
- `scheduler_config`: Scheduling parameters
- `speculative_config`: Spec decode settings
- `lora_config`: LoRA adapter settings

## Performance Optimizations

### 1. CUDA Graph Capture
- Pre-captured graphs for common batch sizes
- Reduces kernel launch overhead

### 2. Continuous Batching
- No padding between requests
- Dynamic batch formation

### 3. PagedAttention
- Block-based KV cache
- Efficient memory utilization

### 4. Prefix Caching
- Hash-based block identification
- Cross-request sharing

### 5. Async Scheduling
- Overlap scheduling with execution
- Pipeline parallelism support

## Differences from V0

| Aspect | V0 | V1 |
|--------|-----|-----|
| Process Model | Single process | Multi-process with ZMQ |
| Scheduler | Sequence-based | Token-based iteration |
| KV Cache | Paged attention | Block pool + prefix caching |
| Spec Decode | Limited | Full support (Eagle, Medusa, etc.) |
| Data Parallel | Basic | Coordinator-based with waves |
| Multimodal | Separate path | Integrated encoder cache |

## File Structure Summary

```
vllm/v1/
├── engine/
│   ├── async_llm.py         # Async engine interface
│   ├── llm_engine.py        # Sync engine interface
│   ├── core.py              # EngineCore implementation
│   ├── core_client.py       # Client for EngineCore
│   ├── processor.py         # Input preprocessing
│   ├── output_processor.py  # Output handling
│   └── detokenizer.py       # Detokenization
├── core/
│   ├── sched/
│   │   ├── scheduler.py     # Main scheduler
│   │   ├── interface.py     # Scheduler interface
│   │   └── output.py        # Scheduler outputs
│   ├── kv_cache_manager.py  # KV cache management
│   ├── block_pool.py        # Block allocation
│   └── encoder_cache_manager.py
├── executor/
│   ├── abstract.py          # Executor interface
│   ├── multiproc_executor.py
│   ├── ray_executor.py
│   └── uniproc_executor.py
├── worker/
│   ├── gpu_worker.py        # GPU worker
│   ├── gpu_model_runner.py  # Model execution
│   ├── gpu_input_batch.py   # Batch management
│   └── worker_base.py       # Base classes
├── attention/
│   └── backends/            # Attention implementations
├── sample/
│   ├── sampler.py           # Token sampling
│   └── metadata.py          # Sampling metadata
├── spec_decode/
│   ├── eagle.py             # Eagle proposer
│   ├── medusa.py            # Medusa proposer
│   └── ngram_proposer.py    # N-gram proposer
├── structured_output/       # Constrained decoding
├── request.py               # Request model
├── outputs.py               # Output definitions
└── kv_cache_interface.py    # KV cache specs
```

## Conclusion

vLLM V1 represents a mature, production-ready architecture for LLM inference. Key strengths include:

1. **Scalability**: Multi-process design with flexible parallelism
2. **Performance**: CUDA graphs, prefix caching, continuous batching
3. **Flexibility**: Pluggable executors, attention backends, spec decode methods
4. **Observability**: Comprehensive metrics and profiling support
5. **Extensibility**: Clean interfaces for custom components

The architecture is designed to handle high-throughput, low-latency inference workloads while supporting advanced features like speculative decoding, disaggregated inference, and multimodal inputs.
