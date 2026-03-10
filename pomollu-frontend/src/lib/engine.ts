import type { RenderedCard } from '$lib/types.js';
import type { TypeDesc } from '$lib/type-parser.js';
import {
	typecheck as wasmTypecheck,
	typecheck_with_types as wasmTypecheckWithTypes,
	typecheck_with_tail as wasmTypecheckWithTail,
	analyze as wasmAnalyze,
	analyze_with_types as wasmAnalyzeWithTypes,
	analyze_with_known as wasmAnalyzeWithKnown,
	analyze_with_tail as wasmAnalyzeWithTail,
	evaluate as wasmEvaluate,
	typecheck_nodes as wasmTypecheckNodes,
	ChatSession as WasmChatSession
} from '$lib/wasm/pomollu_engine.js';

export type CheckResult = { ok: true } | { ok: false; message: string };

export type ContextKeyInfo = { name: string; type: TypeDesc; status: 'eager' | 'lazy' | 'pruned' };

export type AnalyzeResult = {
	ok: true;
	errors: [];
	context_keys: ContextKeyInfo[];
	tail_type: TypeDesc;
} | {
	ok: false;
	errors: string[];
	context_keys: [];
	tail_type: null;
};

export function typecheck(source: string, mode: 'script' | 'template'): CheckResult {
	return wasmTypecheck(source, mode) as CheckResult;
}

export function typecheckWithTypes(
	source: string,
	mode: 'script' | 'template',
	contextTypes: Record<string, TypeDesc>
): CheckResult {
	return wasmTypecheckWithTypes(source, mode, JSON.stringify(contextTypes)) as CheckResult;
}

export function typecheckWithTail(
	source: string,
	mode: 'script' | 'template',
	contextTypes: Record<string, TypeDesc>,
	expectedTailType: TypeDesc
): CheckResult {
	return wasmTypecheckWithTail(source, mode, JSON.stringify(contextTypes), JSON.stringify(expectedTailType)) as CheckResult;
}

export function analyze(source: string, mode: 'script' | 'template'): AnalyzeResult {
	return wasmAnalyze(source, mode) as AnalyzeResult;
}

export function analyzeWithTypes(
	source: string,
	mode: 'script' | 'template',
	contextTypes: Record<string, TypeDesc>
): AnalyzeResult {
	return wasmAnalyzeWithTypes(source, mode, JSON.stringify(contextTypes)) as AnalyzeResult;
}

export function analyzeWithKnown(
	source: string,
	mode: 'script' | 'template',
	contextTypes: Record<string, TypeDesc>,
	knownScripts: Record<string, string>
): AnalyzeResult {
	return wasmAnalyzeWithKnown(source, mode, JSON.stringify(contextTypes), JSON.stringify(knownScripts)) as AnalyzeResult;
}

export function analyzeWithTail(
	source: string,
	mode: 'script' | 'template',
	contextTypes: Record<string, TypeDesc>,
	expectedTailType: TypeDesc
): AnalyzeResult {
	return wasmAnalyzeWithTail(source, mode, JSON.stringify(contextTypes), JSON.stringify(expectedTailType)) as AnalyzeResult;
}

export type WebNode = {
	name: string;
	kind: string;
	api: string;
	model: string;
	temperature: number;
	topP: number | null;
	topK: number | null;
	grounding: boolean;
	maxTokens: { input: number; output: number };
	selfSpec: { initialValue?: string };
	strategy:
		| { mode: 'always' }
		| { mode: 'once-per-turn' }
		| { mode: 'if-modified'; key: string }
		| { mode: 'history'; historyBind: string };
	retry: number;
	assert: string;
	messages: (
		| { kind: 'block'; role: string; template: string }
		| { kind: 'iterator'; iterator: string; role?: string; slice?: number[]; tokenBudget?: { priority: number; min?: number; max?: number } }
	)[];
	tools: { name: string; description: string; node: string; params: { name: string; type: string }[] }[];
};

export type NodeFieldErrors = Record<string, string>;

export type TypecheckNodesResult = {
	contextTypes: Record<string, TypeDesc>;
	nodeLocals: Record<string, { raw: TypeDesc; self: TypeDesc }>;
	nodeErrors: Record<string, NodeFieldErrors>;
} | { error: string };

export function typecheckNodes(
	nodes: WebNode[],
	injectedTypes: Record<string, TypeDesc>
): TypecheckNodesResult {
	return wasmTypecheckNodes(JSON.stringify(nodes), JSON.stringify(injectedTypes)) as
		TypecheckNodesResult;
}

export async function evaluate(
	source: string,
	mode: 'script' | 'template',
	context: Record<string, unknown>
): Promise<unknown> {
	return wasmEvaluate(source, mode, JSON.stringify(context));
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
	api?: string;
	model?: string;
	temperature?: number;
	top_p?: number | null;
	top_k?: number | null;
	grounding?: boolean;
	max_tokens?: { input?: number; output?: number };
	messages?: MessageConfig[];
	tools?: { name: string; description: string; node: string; params: Record<string, string> }[];

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
	api: string;
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
		storage: unknown | null = null,
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

	exportStorage(): unknown {
		if (this._freed) return {};
		return this.inner.export_storage();
	}

	exportStorageJson(): Record<string, unknown> {
		if (this._freed) return {};
		return this.inner.export_storage_json() as Record<string, unknown>;
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
