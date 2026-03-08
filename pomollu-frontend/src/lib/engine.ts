import type { RenderedCard } from '$lib/types.js';
import {
	typecheck as wasmTypecheck,
	typecheck_with_types as wasmTypecheckWithTypes,
	typecheck_with_tail as wasmTypecheckWithTail,
	analyze as wasmAnalyze,
	analyze_with_types as wasmAnalyzeWithTypes,
	analyze_with_tail as wasmAnalyzeWithTail,
	evaluate as wasmEvaluate,
	typecheck_nodes as wasmTypecheckNodes,
	ChatSession as WasmChatSession
} from '$lib/wasm/pomollu_engine.js';

export type CheckResult = { ok: true } | { ok: false; message: string };

export type ContextKeyInfo = { name: string; type: string };

export type AnalyzeResult = {
	ok: true;
	errors: [];
	context_keys: ContextKeyInfo[];
	tail_type: string;
} | {
	ok: false;
	errors: string[];
	context_keys: [];
	tail_type: '';
};

export function typecheck(source: string, mode: 'script' | 'template'): CheckResult {
	return wasmTypecheck(source, mode) as CheckResult;
}

export function typecheckWithTypes(
	source: string,
	mode: 'script' | 'template',
	contextTypes: Record<string, string>
): CheckResult {
	return wasmTypecheckWithTypes(source, mode, JSON.stringify(contextTypes)) as CheckResult;
}

export function typecheckWithTail(
	source: string,
	mode: 'script' | 'template',
	contextTypes: Record<string, string>,
	expectedTailType: string
): CheckResult {
	return wasmTypecheckWithTail(source, mode, JSON.stringify(contextTypes), expectedTailType) as CheckResult;
}

export function analyze(source: string, mode: 'script' | 'template'): AnalyzeResult {
	return wasmAnalyze(source, mode) as AnalyzeResult;
}

export function analyzeWithTypes(
	source: string,
	mode: 'script' | 'template',
	contextTypes: Record<string, string>
): AnalyzeResult {
	return wasmAnalyzeWithTypes(source, mode, JSON.stringify(contextTypes)) as AnalyzeResult;
}

export function analyzeWithTail(
	source: string,
	mode: 'script' | 'template',
	contextTypes: Record<string, string>,
	expectedTailType: string
): AnalyzeResult {
	return wasmAnalyzeWithTail(source, mode, JSON.stringify(contextTypes), expectedTailType) as AnalyzeResult;
}

export type WebNode = {
	name: string;
	kind: string;
	api: string;
	model: string;
	temperature: number;
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
	contextTypes: Record<string, string>;
	nodeLocals: Record<string, { raw: string; self: string }>;
	nodeErrors: Record<string, NodeFieldErrors>;
} | { error: string };

export function typecheckNodes(
	nodes: WebNode[],
	injectedTypes: Record<string, string>
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
	context?: Record<string, { type?: string }>;
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
	max_tokens?: { input?: number; output?: number };
	messages?: MessageConfig[];
	tools?: { name: string; description: string; node: string; params: Record<string, string> }[];

	// Plain/Expr-specific
	template?: string;
	output_ty?: string;
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

	private constructor(inner: WasmChatSession) {
		this.inner = inner;
	}

	/** True while turn() is running (&mut self held). UI should lock during this. */
	get busy(): boolean {
		return this._busy;
	}

	static async create(
		config: SessionConfig,
		storage: unknown | null = null,
	): Promise<ChatSession> {
		const inner = await WasmChatSession.create(
			JSON.stringify(config),
			storage,
		);
		return new ChatSession(inner);
	}

	async turn(resolver: ResolverFn): Promise<unknown> {
		this._busy = true;
		try {
			return await this.inner.turn(resolver);
		} finally {
			this._busy = false;
			if (this._pendingFree) {
				this.inner.free();
			}
		}
	}

	exportStorage(): unknown {
		return this.inner.export_storage();
	}

	exportStorageJson(): Record<string, unknown> {
		return this.inner.export_storage_json() as Record<string, unknown>;
	}

	turnCount(): number {
		return this.inner.turn_count();
	}

	async displayListLen(iteratorScript: string): Promise<number> {
		return this.inner.display_list_len(iteratorScript);
	}

	async renderDisplay(
		iteratorScript: string,
		entriesJson: string,
		index: number
	): Promise<RenderedCard[]> {
		return this.inner.render_display(iteratorScript, entriesJson, index) as RenderedCard[];
	}

	async renderStatic(template: string): Promise<RenderedCard[]> {
		return this.inner.render_static(template) as RenderedCard[];
	}

	/** Safe to call anytime — defers actual free until turn() completes if busy. */
	free(): void {
		if (this._busy) {
			this._pendingFree = true;
		} else {
			this.inner.free();
		}
	}
}
