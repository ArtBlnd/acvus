import type { RenderedCard } from '$lib/types.js';
import type { TypeDesc } from '$lib/type-parser.js';
import type {
	TypeDesc as WasmTypeDesc,
	AnalyzeResult as WasmAnalyzeResult,
	TypecheckNodesResult as WasmTypecheckNodesResult,
	NodeLocalTypes as WasmNodeLocalTypes,
	EngineError,
	NodeErrors as WasmNodeErrors,
	StorageSnapshot,
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

export type { EngineError, StorageSnapshot };

export type NodeErrors = {
	initialValue: EngineError[];
	historyBind: EngineError[];
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

function mapToRecord<V>(mapOrObj: Map<string, V> | Record<string, V>): Record<string, V> {
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

const VALID_PRIMITIVES = new Set(['String', 'Int', 'Float', 'Bool']);

function convertTypeDesc(wasm: WasmTypeDesc): TypeDesc {
	switch (wasm.kind) {
		case 'primitive': {
			if (!VALID_PRIMITIVES.has(wasm.name)) {
				throw new Error(`unknown primitive type from WASM: "${wasm.name}"`);
			}
			return { kind: 'primitive', name: wasm.name as 'String' | 'Int' | 'Float' | 'Bool' };
		}
		case 'option':
			return { kind: 'option', inner: convertTypeDesc(wasm.inner) };
		case 'list':
			return { kind: 'list', elem: convertTypeDesc(wasm.elem) };
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
	strategy:
		| { mode: 'always' }
		| { mode: 'once-per-turn' }
		| { mode: 'if-modified'; key: string }
		| { mode: 'history'; historyBind: string };
	retry: number;
	assert: string;
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
		maxTokens: { input: number; output: number };
		messages: (
			| { kind: 'block'; role: string; template: string }
			| { kind: 'iterator'; iterator: string; role?: string; slice?: number[]; tokenBudget?: { priority: number; min?: number; max?: number } }
		)[];
		tools: { name: string; description: string; node: string; params: { name: string; type: string }[] }[];
	}
	| {
		kind: 'expr';
		exprSource: string;
		initialValue?: string;
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
		historyBind: raw.historyBind,
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

export type StrategyConfig =
	| { mode: 'always' }
	| { mode: 'once-per-turn' }
	| { mode: 'if-modified'; key: string }
	| { mode: 'history'; history_bind: string };

export type NodeConfig = {
	name: string;
	kind: string;
	initial_value?: string;
	strategy: StrategyConfig;
	retry: number;
	assert_script?: string;

	// LLM-specific
	provider?: string;
	api?: 'openai' | 'anthropic' | 'google';
	model?: string;
	temperature?: number;
	top_p?: number | null;
	top_k?: number | null;
	grounding?: boolean;
	max_tokens?: { input?: number; output?: number };
	messages?: MessageConfig[];
	tools?: { name: string; description: string; node: string; params: Record<string, string> }[];

	// Function node
	is_function?: boolean;
	fn_params?: { name: string; type: string }[];

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
	private _busy = false;
	private _pendingFree = false;
	private _freed = false;
	private _crashed = false;

	private constructor(inner: WasmChatSession) {
		this.inner = inner;
	}

	/** True while turn() is running (&mut self held by WASM). */
	get busy(): boolean {
		return this._busy;
	}

	/** True after inner WASM object has been freed. */
	get freed(): boolean {
		return this._freed;
	}

	/** True after a WASM panic/trap — session is unusable. */
	get crashed(): boolean {
		return this._crashed;
	}

	static async create(
		config: SessionConfig,
		storage: StorageSnapshot | null = null,
		onStorageChange: (key: string, value: unknown) => void,
	): Promise<ChatSession> {
		const inner = await WasmChatSession.create(
			JSON.stringify(config),
			storage,
			onStorageChange,
		);
		return new ChatSession(inner);
	}

	async turn(resolver: ResolverFn): Promise<unknown> {
		if (this._freed) throw new Error('ChatSession already freed');
		if (this._crashed) throw new Error('ChatSession crashed — recreate session');
		this._busy = true;
		try {
			// WASM panic=abort causes a WebAssembly.RuntimeError that escapes
			// the wasm-bindgen-futures executor, leaving the inner Promise
			// permanently pending. Catch it via window 'error' and force-reject.
			return await new Promise((resolve, reject) => {
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
					(v) => { settled = true; resolve(v); },
					(e) => { settled = true; reject(e); },
				).finally(() => {
					window.removeEventListener('error', onError);
				});
			});
		} finally {
			this._busy = false;
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

	exportStorage(): StorageSnapshot {
		if (this._freed) return {} as StorageSnapshot;
		return this.inner.export_storage();
	}

	turnCount(): number {
		if (this._freed) return 0;
		return this.inner.turn_count();
	}

	async displayListLen(iteratorScript: string): Promise<number> {
		if (this._freed || this._busy) return 0;
		return this.inner.display_list_len(iteratorScript);
	}

	async renderDisplay(
		iteratorScript: string,
		entriesJson: string,
		index: number
	): Promise<RenderedCard[]> {
		if (this._freed || this._busy) return [];
		return this.inner.render_display(iteratorScript, entriesJson, index) as unknown as RenderedCard[];
	}

	async renderStatic(template: string): Promise<RenderedCard[]> {
		if (this._freed || this._busy) return [];
		return this.inner.render_static(template) as unknown as RenderedCard[];
	}

	/**
	 * Request free. If turn() is in progress, defers until finishTurn().
	 * If not busy, frees immediately.
	 */
	free(): void {
		if (this._freed) return;
		if (this._busy) {
			this._pendingFree = true;
		} else {
			this._freed = true;
			this.inner.free();
		}
	}
}
