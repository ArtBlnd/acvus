import type { RenderedCard } from '$lib/types.js';
import type { TypeDesc } from '$lib/type-parser.js';
import type {
	TypeDesc as WasmTypeDesc,
	TypecheckNodesResult as WasmTypecheckNodesResult,
	NodeLocalTypes as WasmNodeLocalTypes,
	EngineError,
	NodeErrors as WasmNodeErrors,
	DiagnosticsResult as WasmDiagnosticsResult,
	ContextKeysResult as WasmContextKeysResult,
	ContextKey as WasmContextKey,
	WasmScope,
	WasmKnownValues,
} from '$lib/wasm/pomollu_engine.js';
import {
	evaluate as wasmEvaluate,
	ChatSession as WasmChatSession,
	LanguageSession as WasmLanguageSession,
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
		case 'sequence':
			return { kind: 'sequence', elem: convertTypeDesc(wasm.elem), origin: wasm.origin };
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
// Context key info — returned by LanguageSession.contextKeys()
// ---------------------------------------------------------------------------

export type ContextKeyInfo = { name: string; type: TypeDesc; status: 'eager' | 'lazy' | 'pruned' };

// ---------------------------------------------------------------------------
// TypecheckNodes result types — used by LanguageSession.rebuildNodes()
// ---------------------------------------------------------------------------

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
			| { kind: 'sequence'; bind: string }
			| { kind: 'patch'; bind: string };
		initialValue?: string;
		retry: number;
		assert: string;
	};
	isFunction: boolean;
	fnParams: { name: string; type: string }[];
};

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

function convertContextKey(raw: WasmContextKey): ContextKeyInfo {
	return {
		name: raw.name,
		type: convertTypeDesc(raw.type),
		status: raw.status,
	};
}

// ---------------------------------------------------------------------------
// Evaluate
// ---------------------------------------------------------------------------

export async function evaluate(
	source: string,
	mode: 'script' | 'template',
	context?: Record<string, unknown>
): Promise<unknown> {
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
	asset_store_name?: string;
};

export type ExecutionConfig =
	| { mode: 'always' }
	| { mode: 'once-per-turn' }
	| { mode: 'if-modified'; key: string };

export type PersistencyConfig =
	| { kind: 'ephemeral' }
	| { kind: 'snapshot' }
	| { kind: 'sequence'; bind: string }
	| { kind: 'patch'; bind: string };

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
	is_function?: boolean;
	fn_params?: { name: string; type: string; description?: string }[];
	template?: string;
	output_ty?: TypeDesc;
	iterator?: string;
	sources?: { name: string; node: string }[];
	unordered?: boolean;
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

	get busy(): boolean {
		return this._exclusive;
	}

	get freed(): boolean {
		return this._freed;
	}

	get crashed(): boolean {
		return this._crashed;
	}

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

	async startEvaluate(nodeName: string, noExecute: boolean, resolver: ResolverFn): Promise<void> {
		if (this._freed) throw new Error('ChatSession already freed');
		if (this._crashed) throw new Error('ChatSession crashed — recreate session');
		await this.acquireExclusive();
		try {
			await this.inner.start_evaluate(nodeName, noExecute, resolver);
		} catch (e) {
			this.releaseExclusive();
			throw e;
		}
	}

	async evaluateNext(resolver: ResolverFn): Promise<unknown | null> {
		if (this._freed) throw new Error('ChatSession already freed');
		if (this._crashed) throw new Error('ChatSession crashed — recreate session');
		return await this.inner.evaluate_next(resolver);
	}

	cancelEvaluate(): void {
		if (this._freed) return;
		this.inner.cancel_evaluate();
	}

	finishEvaluate(): void {
		this.releaseExclusive();
		if (this._pendingFree && !this._freed) {
			this._freed = true;
			this.inner.free();
		}
	}

	/** @deprecated Use finishEvaluate() */
	finishTurn(): void {
		this.finishEvaluate();
	}

	async undo(): Promise<void> {
		if (this._freed) return;
		await this.acquireExclusive();
		try {
			await this.inner.undo();
		} finally {
			this.releaseExclusive();
		}
	}

	async goto(id: string): Promise<void> {
		if (this._freed) return;
		await this.acquireExclusive();
		try {
			await this.inner.goto(id);
		} finally {
			this.releaseExclusive();
		}
	}

	async turnCount(): Promise<number> {
		return this.withReadGuard(0, () => this.inner.turn_count());
	}

	tree(): TreeView | null {
		if (this._freed || this._exclusive) return null;
		return this.inner.tree() as unknown as TreeView;
	}

	stateAt(id: string): StorageViewResult | null {
		if (this._freed || this._exclusive) return null;
		return this.inner.state_at(id) as unknown as StorageViewResult;
	}

	visibleState(): StorageViewResult | null {
		if (this._freed || this._exclusive) return null;
		return this.inner.visible_state() as unknown as StorageViewResult;
	}

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

// ---------------------------------------------------------------------------
// Completion types
// ---------------------------------------------------------------------------

export type CompletionKind = 'context' | 'builtin' | 'keyword';

export type CompletionItem = {
	label: string;
	kind: CompletionKind;
	detail: string;
	insertText: string;
};

// ---------------------------------------------------------------------------
// Document scope — provided (engine) vs user (params)
// ---------------------------------------------------------------------------

export type DocScope = {
	/** Engine-provided types — context_keys excludes these. */
	provided: Record<string, TypeDesc>;
	/** User-declared param types — context_keys includes these. */
	user: Record<string, TypeDesc>;
};

// ---------------------------------------------------------------------------
// LanguageSession — document-centric LSP API
// ---------------------------------------------------------------------------

export class LanguageSession {
	private inner: WasmLanguageSession;
	private _freed = false;

	private constructor(inner: WasmLanguageSession) {
		this.inner = inner;
	}

	get freed(): boolean {
		return this._freed;
	}

	static create(): LanguageSession {
		return new LanguageSession(new WasmLanguageSession());
	}

	// --- Document management ---

	/** Open a new document. Returns its numeric ID. */
	open(source: string, mode: 'script' | 'template', scope: DocScope): number {
		if (this._freed) throw new Error('LanguageSession already freed');
		return this.inner.open(source, mode, scope as unknown as WasmScope);
	}

	/** Update a document's source. Invalidates all caches. */
	updateSource(docId: number, source: string): void {
		if (this._freed) return;
		this.inner.update_source(docId, source);
	}

	/** Update a document's scope. Invalidates typecheck caches. */
	updateScope(docId: number, scope: DocScope): void {
		if (this._freed) return;
		this.inner.update_scope(docId, scope as unknown as WasmScope);
	}

	/** Update both source and scope atomically. */
	update(docId: number, source: string, scope: DocScope): void {
		if (this._freed) return;
		this.inner.update(docId, source, scope as unknown as WasmScope);
	}

	/** Close a document. */
	close(docId: number): void {
		if (this._freed) return;
		this.inner.close(docId);
	}

	/** Bind a document to a node field (inference unit). */
	bindDocToNode(docId: number, nodeName: string, field: string, fieldIndex?: number): void {
		if (this._freed) return;
		this.inner.bind_doc_to_node(docId, nodeName, field, fieldIndex ?? null);
	}

	// --- Queries ---

	/** Get diagnostics for a document. */
	diagnostics(docId: number): { ok: boolean; errors: EngineError[] } {
		if (this._freed) return { ok: true, errors: [] };
		const raw = this.inner.diagnostics(docId) as WasmDiagnosticsResult;
		return { ok: raw.ok, errors: raw.errors };
	}

	/** Discover context keys for a document. */
	contextKeys(docId: number, knownValues?: Record<string, string>): ContextKeyInfo[] {
		if (this._freed) return [];
		const raw = this.inner.context_keys(
			docId,
			{ values: knownValues ?? {} } as unknown as WasmKnownValues,
		) as WasmContextKeysResult;
		return raw.keys.map(convertContextKey);
	}

	/** Get completions at cursor position. */
	completions(docId: number, cursor: number): CompletionItem[] {
		if (this._freed) return [];
		const result = this.inner.completions(docId, cursor) as { items: CompletionItem[] };
		return result.items;
	}

	// --- Node-level rebuild ---

	/** Full node-level rebuild — typecheck all nodes. */
	rebuildNodes(nodes: WebNode[], injectedTypes: Record<string, TypeDesc>): TypecheckNodesResult {
		if (this._freed) throw new Error('LanguageSession already freed');
		const raw = this.inner.rebuild_nodes({ nodes, injectedTypes }) as WasmTypecheckNodesResult;
		return convertTypecheckNodesResult(raw);
	}

	free(): void {
		if (this._freed) return;
		this._freed = true;
		this.inner.free();
	}
}
