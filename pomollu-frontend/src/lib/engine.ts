import type { RenderedCard } from '$lib/types.js';
import type { TypeDesc } from '$lib/type-parser.js';
import type {
	TypeDesc as WasmTypeDesc,
	AnalyzeResult as WasmAnalyzeResult,
	TypecheckNodesResult as WasmTypecheckNodesResult,
	NodeLocalTypes as WasmNodeLocalTypes,
	EngineError,
	NodeErrors as WasmNodeErrors,
} from '$lib/wasm/pomollu_engine.js';
import {
	analyze as wasmAnalyze,
	typecheck as wasmTypecheck,
	evaluate as wasmEvaluate,
	typecheck_nodes as wasmTypecheckNodes,
	ChatSession as WasmChatSession,
} from '$lib/wasm/pomollu_engine.js';

// ---------------------------------------------------------------------------
// Re-exports — WASM types that consumers use directly
// ---------------------------------------------------------------------------

export type { EngineError };

// ---------------------------------------------------------------------------
// Tree / Turn API types
// ---------------------------------------------------------------------------

export type TurnNode = {
	uuid: string;
	parent: string | null;
	depth: number;
};

export type TreeView = {
	nodes: TurnNode[];
	cursor: string;
};

export type TurnResultWithNode = {
	value: unknown;
	turn: TurnNode;
};

// ---------------------------------------------------------------------------
// History API types (mirrors Rust StorageView via Tsify)
// ---------------------------------------------------------------------------

export type StorageViewResult = {
	cursor: string;
	depth: number;
	entries: Map<string, unknown> | Record<string, unknown>;
};

export type NodeErrors = {
	initialValue: EngineError[];
	bind: EngineError[];
	ifModifiedKey: EngineError[];
	assert: EngineError[];
	messages: Record<string, EngineError[]>;
	exprSource: EngineError[];
};

// ---------------------------------------------------------------------------
// Map → Record helper
// ---------------------------------------------------------------------------
// Tsify serializes FxHashMap as JS Map, not plain object.
// We need to convert to Record for TS consumers.

export function mapToRecord<V>(mapOrObj: Map<string, V> | Record<string, V>): Record<string, V> {
	if (mapOrObj instanceof Map) {
		const rec: Record<string, V> = {};
		for (const [k, v] of mapOrObj) rec[k] = v;
		return rec;
	}
	return mapOrObj;
}

// ---------------------------------------------------------------------------
// WASM TypeDesc → type-parser TypeDesc conversion
// ---------------------------------------------------------------------------

const VALID_PRIMITIVES = new Set(['string', 'int', 'float', 'bool']);

function convertTypeDesc(wasm: WasmTypeDesc): TypeDesc {
	switch (wasm.kind) {
		case 'primitive': {
			if (!VALID_PRIMITIVES.has(wasm.name)) {
				throw new Error(`unknown primitive type from WASM: "${wasm.name}"`);
			}
			return { kind: 'primitive', name: wasm.name as 'string' | 'int' | 'float' | 'bool' };
		}
		case 'option':
			return { kind: 'option', inner: convertTypeDesc(wasm.inner) };
		case 'list':
			return { kind: 'list', elem: convertTypeDesc(wasm.elem) };
		case 'deque':
			return { kind: 'deque', elem: convertTypeDesc(wasm.elem), origin: wasm.origin };
		case 'object':
			return {
				kind: 'object',
				fields: wasm.fields.map((f) => ({ name: f.name, type: convertTypeDesc(f.type) })),
			};
		case 'enum':
			return {
				kind: 'enum',
				name: wasm.name,
				variants: wasm.variants.map((v) => ({
					tag: v.tag,
					hasPayload: v.hasPayload,
					payloadType: v.payloadType ? convertTypeDesc(v.payloadType) : undefined,
				})),
			};
		case 'unsupported':
			return { kind: 'unsupported', raw: wasm.raw };
	}
}

// ---------------------------------------------------------------------------
// EngineError[] formatting helper
// ---------------------------------------------------------------------------

export function formatErrors(errors: EngineError[] | undefined): string {
	if (!errors || errors.length === 0) return '';
	return errors.map((e) => e.message).join('\n');
}

// ---------------------------------------------------------------------------
// CheckResult — no TypeDesc, pass-through
// ---------------------------------------------------------------------------

export type CheckResult = { ok: boolean; errors: EngineError[] };

export function typecheck(source: string, mode: 'script' | 'template'): CheckResult {
	return wasmTypecheck({ source, mode });
}

export function typecheckWithTypes(
	source: string,
	mode: 'script' | 'template',
	contextTypes: Record<string, TypeDesc>
): CheckResult {
	return wasmTypecheck({ source, mode, contextTypes });
}

export function typecheckWithTail(
	source: string,
	mode: 'script' | 'template',
	contextTypes: Record<string, TypeDesc>,
	expectedTail: TypeDesc
): CheckResult {
	return wasmTypecheck({ source, mode, contextTypes, expectedTail });
}

// ---------------------------------------------------------------------------
// AnalyzeResult — convert WASM TypeDesc → type-parser TypeDesc
// ---------------------------------------------------------------------------

export type ContextKeyInfo = { name: string; type: TypeDesc; status: 'eager' | 'lazy' | 'pruned' };

export type AnalyzeResult = {
	ok: boolean;
	errors: EngineError[];
	contextKeys: ContextKeyInfo[];
	tailType: TypeDesc;
};

function convertAnalyzeResult(raw: WasmAnalyzeResult): AnalyzeResult {
	return {
		ok: raw.ok,
		errors: raw.errors,
		contextKeys: raw.contextKeys.map((k) => ({
			name: k.name,
			type: convertTypeDesc(k.type),
			status: k.status,
		})),
		tailType: convertTypeDesc(raw.tailType),
	};
}

export function analyze(source: string, mode: 'script' | 'template'): AnalyzeResult {
	return convertAnalyzeResult(wasmAnalyze({ source, mode }));
}

export function analyzeWithTypes(
	source: string,
	mode: 'script' | 'template',
	contextTypes: Record<string, TypeDesc>
): AnalyzeResult {
	return convertAnalyzeResult(wasmAnalyze({ source, mode, contextTypes }));
}

export function analyzeWithKnown(
	source: string,
	mode: 'script' | 'template',
	contextTypes: Record<string, TypeDesc>,
	knownValues: Record<string, string>
): AnalyzeResult {
	return convertAnalyzeResult(wasmAnalyze({ source, mode, contextTypes, knownValues }));
}

export function analyzeWithTail(
	source: string,
	mode: 'script' | 'template',
	contextTypes: Record<string, TypeDesc>,
	expectedTail: TypeDesc
): AnalyzeResult {
	return convertAnalyzeResult(wasmAnalyze({ source, mode, contextTypes, expectedTail }));
}

// ---------------------------------------------------------------------------
// TypecheckNodes — convert WASM TypeDesc → type-parser TypeDesc
// ---------------------------------------------------------------------------

type WebNodeShared = {
	name: string;
	strategy: {
		execution:
			| { mode: 'always' }
			| { mode: 'once-per-turn' }
			| { mode: 'if-modified'; key: string };
		persistency:
			| { kind: 'ephemeral' }
			| { kind: 'snapshot' }
			| { kind: 'deque'; bind: string }
			| { kind: 'diff'; bind: string };
		initialValue?: string;
		retry: number;
		assert: string;
	};
	isFunction: boolean;
	fnParams: { name: string; type: string }[];
};

export type WebNode = WebNodeShared & (
	| {
		kind: 'llm';
		api?: 'openai' | 'anthropic' | 'google';
		model: string;
		temperature: number;
		topP: number | null;
		topK: number | null;
		grounding: boolean;
		thinking: import('./types.js').ThinkingConfig | null;
		maxTokens: { input: number; output: number };
		messages: (
			| { kind: 'block'; role: string; template: string }
			| { kind: 'iterator'; iterator: string; role?: string; slice?: number[]; tokenBudget?: { priority: number; min?: number; max?: number } }
		)[];
		tools: { name: string; description: string; node: string; params: { name: string; type: string; description?: string }[] }[];
	}
	| {
		kind: 'expr';
		exprSource: string;
	}
	| {
		kind: 'plain';
	}
);

export type TypecheckNodesResult = {
	envErrors: EngineError[];
	contextTypes: Record<string, TypeDesc>;
	nodeLocals: Record<string, { raw: TypeDesc; self: TypeDesc }>;
	nodeErrors: Record<string, NodeErrors>;
};

function convertNodeErrors(raw: WasmNodeErrors): NodeErrors {
	return {
		initialValue: raw.initialValue,
		bind: raw.bind,
		ifModifiedKey: raw.ifModifiedKey,
		assert: raw.assert,
		messages: mapToRecord(raw.messages as unknown as Map<string, EngineError[]>),
		exprSource: raw.exprSource ?? [],
	};
}

function convertTypecheckNodesResult(raw: WasmTypecheckNodesResult): TypecheckNodesResult {
	const rawContextTypes = mapToRecord(raw.contextTypes as unknown as Map<string, WasmTypeDesc>);
	const contextTypes: Record<string, TypeDesc> = {};
	for (const [k, v] of Object.entries(rawContextTypes)) {
		contextTypes[k] = convertTypeDesc(v);
	}
	const rawNodeLocals = mapToRecord(raw.nodeLocals as unknown as Map<string, WasmNodeLocalTypes>);
	const nodeLocals: Record<string, { raw: TypeDesc; self: TypeDesc }> = {};
	for (const [k, v] of Object.entries(rawNodeLocals)) {
		nodeLocals[k] = { raw: convertTypeDesc(v.raw), self: convertTypeDesc(v.self) };
	}
	const rawNodeErrors = mapToRecord(raw.nodeErrors as unknown as Map<string, WasmNodeErrors>);
	const nodeErrors: Record<string, NodeErrors> = {};
	for (const [k, v] of Object.entries(rawNodeErrors)) {
		nodeErrors[k] = convertNodeErrors(v);
	}
	return {
		envErrors: raw.envErrors,
		contextTypes,
		nodeLocals,
		nodeErrors,
	};
}

export function typecheckNodes(
	nodes: WebNode[],
	injectedTypes: Record<string, TypeDesc>
): TypecheckNodesResult {
	return convertTypecheckNodesResult(wasmTypecheckNodes({ nodes, injectedTypes }));
}

// ---------------------------------------------------------------------------
// Evaluate
// ---------------------------------------------------------------------------

export async function evaluate(
	source: string,
	mode: 'script' | 'template',
	context?: Record<string, unknown>
): Promise<unknown> {
	// context values are JsConcreteValue at the WASM boundary — opaque to TS callers
	return wasmEvaluate({ source, mode, context: context as never });
}

// ---------------------------------------------------------------------------
// ChatSession — wraps WASM ChatEngine for multi-turn chat
// ---------------------------------------------------------------------------

export type SessionConfig = {
	nodes: NodeConfig[];
	providers: Record<string, ProviderConfig>;
	entrypoint: string;
	context?: Record<string, { type?: TypeDesc }>;
	side_effects?: string[];
};

export type ExecutionConfig =
	| { mode: 'always' }
	| { mode: 'once-per-turn' }
	| { mode: 'if-modified'; key: string };

export type PersistencyConfig =
	| { kind: 'ephemeral' }
	| { kind: 'snapshot' }
	| { kind: 'deque'; bind: string }
	| { kind: 'diff'; bind: string };

export type StrategyConfig = {
	execution: ExecutionConfig;
	persistency: PersistencyConfig;
	initial_value?: string;
	retry: number;
	assert_script?: string;
};

export type NodeConfig = {
	name: string;
	kind: string;
	strategy: StrategyConfig;

	// LLM-specific
	provider?: string;
	api?: 'openai' | 'anthropic' | 'google';
	model?: string;
	temperature?: number;
	top_p?: number | null;
	top_k?: number | null;
	grounding?: boolean;
	thinking?: import('./types.js').ThinkingConfig | null;
	max_tokens?: { input?: number; output?: number };
	messages?: MessageConfig[];
	tools?: { name: string; description: string; node: string; params: { name: string; type: string; description?: string }[] }[];

	// Function node
	is_function?: boolean;
	fn_params?: { name: string; type: string; description?: string }[];

	// Plain/Expr-specific
	template?: string;
	output_ty?: TypeDesc;
};

export type MessageConfig = {
	role?: string;
	template?: string;
	inline_template?: string;
	iterator?: string;
	slice?: number[];
	token_budget?: { priority: number; min?: number; max?: number };
};

export type ProviderConfig = {
	api: 'openai' | 'anthropic' | 'google';
	endpoint: string;
	api_key: string;
};

export type ResolverFn = (key: string) => string | Promise<string>;

export class ChatSession {
	private inner: WasmChatSession;
	private _exclusive = false;
	private _inflightReads = 0;
	private _drainResolve: (() => void) | null = null;
	private _pendingFree = false;
	private _freed = false;
	private _crashed = false;

	private constructor(inner: WasmChatSession) {
		this.inner = inner;
	}

	/** True while an exclusive operation (turn/undo/goto) holds the lock. */
	get busy(): boolean {
		return this._exclusive;
	}

	/** True after inner WASM object has been freed. */
	get freed(): boolean {
		return this._freed;
	}

	/** True after a WASM panic/trap — session is unusable. */
	get crashed(): boolean {
		return this._crashed;
	}

	// -----------------------------------------------------------------------
	// Lock primitives
	// -----------------------------------------------------------------------

	/**
	 * Acquire exclusive lock. Sets the flag immediately to reject new reads,
	 * then waits for any in-flight read operations to drain.
	 */
	private async acquireExclusive(): Promise<void> {
		if (this._exclusive) throw new Error('Session is busy');
		this._exclusive = true;
		if (this._inflightReads > 0) {
			await new Promise<void>((resolve) => {
				this._drainResolve = resolve;
			});
		}
	}

	private releaseExclusive(): void {
		this._exclusive = false;
	}

	/**
	 * Run fn under a shared read guard. If exclusive lock is held or session
	 * is freed/crashed, returns fallback immediately without touching WASM.
	 */
	private async withReadGuard<T>(fallback: T, fn: () => Promise<T>): Promise<T> {
		if (this._freed || this._exclusive) return fallback;
		this._inflightReads++;
		try {
			return await fn();
		} finally {
			this._inflightReads--;
			if (this._inflightReads === 0 && this._drainResolve) {
				const resolve = this._drainResolve;
				this._drainResolve = null;
				resolve();
			}
		}
	}

	// -----------------------------------------------------------------------
	// Construction
	// -----------------------------------------------------------------------

	static async create(
		config: SessionConfig,
		sessionId: string,
	): Promise<ChatSession> {
		const inner = await WasmChatSession.create(
			JSON.stringify(config),
			sessionId,
		);
		return new ChatSession(inner);
	}

	// -----------------------------------------------------------------------
	// Exclusive operations (turn, undo, goto)
	// -----------------------------------------------------------------------

	async turn(resolver: ResolverFn): Promise<TurnResultWithNode> {
		if (this._freed) throw new Error('ChatSession already freed');
		if (this._crashed) throw new Error('ChatSession crashed — recreate session');
		await this.acquireExclusive();
		try {
			// WASM panic=abort causes a WebAssembly.RuntimeError that escapes
			// the wasm-bindgen-futures executor, leaving the inner Promise
			// permanently pending. Catch it via window 'error' and force-reject.
			return await new Promise<TurnResultWithNode>((resolve, reject) => {
				let settled = false;
				const onError = (e: ErrorEvent) => {
					if (!settled && e.error instanceof WebAssembly.RuntimeError) {
						settled = true;
						this._crashed = true;
						reject(e.error);
					}
				};
				window.addEventListener('error', onError);
				this.inner.turn(resolver).then(
					(v) => { settled = true; resolve(v as unknown as TurnResultWithNode); },
					(e) => { settled = true; reject(e); },
				).finally(() => {
					window.removeEventListener('error', onError);
				});
			});
		} finally {
			this.releaseExclusive();
		}
	}

	/**
	 * Call after turn() and all post-turn work is done.
	 * Actually frees the WASM object if free() was called during turn().
	 */
	finishTurn(): void {
		if (this._pendingFree && !this._freed) {
			this._freed = true;
			this.inner.free();
		}
	}

	/** Undo: move cursor to parent entry. */
	async undo(): Promise<void> {
		if (this._freed) return;
		await this.acquireExclusive();
		try {
			await this.inner.undo();
		} finally {
			this.releaseExclusive();
		}
	}

	/** Navigate to a specific entry by UUID. */
	async goto(id: string): Promise<void> {
		if (this._freed) return;
		await this.acquireExclusive();
		try {
			await this.inner.goto(id);
		} finally {
			this.releaseExclusive();
		}
	}

	// -----------------------------------------------------------------------
	// Read operations (shared, rejected while exclusive lock is held)
	// -----------------------------------------------------------------------

	async turnCount(): Promise<number> {
		return this.withReadGuard(0, () => this.inner.turn_count());
	}

	async displayListLen(iteratorScript: string): Promise<number> {
		return this.withReadGuard(0, () => this.inner.display_list_len(iteratorScript));
	}

	async renderDisplay(
		iteratorScript: string,
		entriesJson: string,
		index: number
	): Promise<RenderedCard[]> {
		return this.withReadGuard([], () =>
			this.inner.render_display(iteratorScript, entriesJson, index) as unknown as Promise<RenderedCard[]>
		);
	}

	async renderStatic(template: string): Promise<RenderedCard[]> {
		return this.withReadGuard([], () =>
			this.inner.render_static(template) as unknown as Promise<RenderedCard[]>
		);
	}

	// -----------------------------------------------------------------------
	// Tree API (sync — no guard needed, just check flags)
	// -----------------------------------------------------------------------

	/** Get the full tree of turn nodes and current cursor. */
	tree(): TreeView | null {
		if (this._freed || this._exclusive) return null;
		return this.inner.tree() as unknown as TreeView;
	}

	// -----------------------------------------------------------------------
	// History API (sync reads)
	// -----------------------------------------------------------------------

	/** Get storage state at a specific entry by UUID. */
	stateAt(id: string): StorageViewResult | null {
		if (this._freed || this._exclusive) return null;
		return this.inner.state_at(id) as unknown as StorageViewResult;
	}

	/** Get visible state at the current cursor. */
	visibleState(): StorageViewResult | null {
		if (this._freed || this._exclusive) return null;
		return this.inner.visible_state() as unknown as StorageViewResult;
	}

	// -----------------------------------------------------------------------
	// Lifecycle
	// -----------------------------------------------------------------------

	/**
	 * Request free. If an exclusive op is in progress, defers until finishTurn().
	 * If not busy, frees immediately.
	 */
	free(): void {
		if (this._freed) return;
		if (this._exclusive) {
			this._pendingFree = true;
		} else {
			this._freed = true;
			this.inner.free();
		}
	}
}
