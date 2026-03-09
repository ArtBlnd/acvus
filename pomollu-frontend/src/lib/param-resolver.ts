import type { ContextKeyInfo, WebNode, TypecheckNodesResult } from './engine.js';
import type { TypeDesc } from './type-parser.js';
import { isUnknownType } from './type-parser.js';
import type { Node, BlockNode, ContextBinding, DisplayEntry, DisplayRegion, ContextParam } from './types.js';
import { isRawBlock, isScriptBlock, BUILTIN_CONTEXT_REFS } from './types.js';
import { collectBlocks, collectNodes } from './block-tree.js';
import { analyzeWithTypes, analyzeWithKnown, typecheckNodes } from './engine.js';

export type ScriptEntry = { source: string; mode: 'script' | 'template' };

export function collectScriptsFromBindings(bindings: ContextBinding[]): ScriptEntry[] {
	return bindings
		.filter((b) => b.script.trim())
		.map((b) => ({ source: b.script, mode: 'script' as const }));
}

export function collectScriptsFromTree(children: BlockNode[]): ScriptEntry[] {
	const scripts: ScriptEntry[] = [];
	for (const block of collectBlocks(children)) {
		if (isRawBlock(block)) {
			scripts.push({ source: block.text, mode: block.mode });
		} else if (isScriptBlock(block)) {
			scripts.push({ source: block.text, mode: 'script' });
		}
	}
	for (const node of collectNodes(children)) {
		collectScriptsFromNode(node, scripts);
	}
	return scripts;
}

function collectScriptsFromNode(node: Node, out: ScriptEntry[]) {
	if (node.selfSpec.initialValue.trim()) out.push({ source: node.selfSpec.initialValue, mode: 'script' });
	if (node.assert.trim() && node.assert !== 'true') out.push({ source: node.assert, mode: 'script' });
	if (node.strategy.mode === 'history' && node.strategy.historyBind.trim()) {
		out.push({ source: node.strategy.historyBind, mode: 'script' });
	}
	if (node.strategy.mode === 'if-modified' && node.strategy.key.trim()) {
		out.push({ source: node.strategy.key, mode: 'script' });
	}
	for (const msg of node.messages) {
		if (msg.kind === 'block' && msg.source.type === 'inline') {
			out.push({ source: msg.source.template, mode: 'template' });
		}
		if (msg.kind === 'iterator' && msg.iterator.trim()) {
			out.push({ source: msg.iterator, mode: 'script' });
		}
		if (msg.kind === 'iterator' && msg.template?.trim()) {
			out.push({ source: msg.template, mode: 'template' });
		}
	}
}

export function collectScriptsFromDisplay(entries: DisplayEntry[]): ScriptEntry[] {
	const scripts: ScriptEntry[] = [];
	for (const entry of entries) {
		if (entry.condition.trim()) scripts.push({ source: entry.condition, mode: 'script' });
		if (entry.template.trim()) scripts.push({ source: entry.template, mode: 'template' });
	}
	return scripts;
}

export function collectScriptsFromRegions(regions: DisplayRegion[]): ScriptEntry[] {
	const scripts: ScriptEntry[] = [];
	for (const region of regions) {
		if (region.kind === 'static' && region.template.trim()) {
			scripts.push({ source: region.template, mode: 'template' });
		} else if (region.kind === 'iterable') {
			if (region.iterator.trim()) scripts.push({ source: region.iterator, mode: 'script' });
			scripts.push(...collectScriptsFromDisplay(region.entries));
		}
	}
	return scripts;
}

export function collectNodeNames(children: BlockNode[]): Set<string> {
	return new Set(collectNodes(children).map((n) => n.name).filter((n) => n));
}

export type CollectParamsResult =
	| { ok: true; keys: ContextKeyInfo[] }
	| { ok: false; errors: string[] };

export function collectUnresolvedParams(opts: {
	scripts: { source: string; mode: 'script' | 'template' }[];
	nodeNames: Set<string>;
	providedKeys: Set<string>;
	contextTypes: Record<string, TypeDesc>;
	knownScripts?: Record<string, string>;
}): CollectParamsResult {
	const seen = new Map<string, { type: TypeDesc; status: 'eager' | 'lazy' | 'pruned' }>();

	const knownEntries = opts.knownScripts ? Object.entries(opts.knownScripts) : [];
	for (const { source, mode } of opts.scripts) {
		if (!source.trim()) continue;
		const useKnown = knownEntries.length > 0;
		const result = useKnown
			? analyzeWithKnown(source, mode, opts.contextTypes, opts.knownScripts!)
			: analyzeWithTypes(source, mode, opts.contextTypes);
		if (!result.ok) {
			console.warn('[collectUnresolvedParams] FAILED script:', mode, source.slice(0, 80), result.errors);
			return { ok: false, errors: result.errors };
		}
		const filteredKeys = result.context_keys.filter(
			(k) => !BUILTIN_CONTEXT_REFS.has(k.name) && !opts.nodeNames.has(k.name) && !opts.providedKeys.has(k.name)
		);
		for (const key of filteredKeys) {
			const existing = seen.get(key.name);
			if (!existing || isUnknownType(existing.type)) {
				seen.set(key.name, { type: key.type, status: key.status });
			} else if (existing.status === 'pruned' && key.status !== 'pruned') {
				// Pruned key found on a live path in another script → upgrade
				seen.set(key.name, { type: existing.type, status: key.status });
			} else if (existing.status === 'lazy' && key.status === 'eager') {
				seen.set(key.name, { type: existing.type, status: 'eager' });
			}
		}
	}

	const keys = Array.from(seen.entries())
		.map(([name, { type, status }]) => ({ name, type, status }))
		.sort((a, b) => a.name.localeCompare(b.name));
	return { ok: true, keys };
}

/**
 * Shared 2-phase analysis pipeline for Prompt/Profile/Bot settings.
 *
 * ── Phase 1: Pure Type Discovery ──────────────────────────────────
 * Analyzes ALL scripts WITHOUT known values.
 * Every branch is live, so every @context ref is discovered with its
 * correctly-inferred type.  This produces the FULL type set.
 * Keys not found here are deactivated (no longer referenced).
 *
 * ── Phase 2: Typecheck + Pruning ──────────────────────────────────
 * 1) Hard typecheck (typecheckNodes) runs with the full type set from
 *    Phase 1, so all branches — including dead ones — compile cleanly.
 * 2) Re-analyze WITH known values to classify dead branches.
 *    Keys that only appear on dead paths are "pruned" and removed from
 *    the unresolved list, hiding them from the UI.
 *
 * IMPORTANT: Phase 1 MUST run without known values.
 * If known values are passed in Phase 1, the engine's partition logic
 * drops known keys on dead branches entirely — they get no type, no
 * status, and Phase 2 typecheck fails with "undefined context" errors.
 */
export type AnalyzeLevelResult =
	| { ok: true; discoveredTypes: Record<string, TypeDesc>; unresolvedKeys: ContextKeyInfo[] }
	| { ok: false; errors: string[]; phase: 'analysis' | 'typecheck' };

export function analyzeLevel(opts: {
	scripts: ScriptEntry[];
	nodeNames: Set<string>;
	providedKeys: Set<string>;
	existingParams: ContextParam[];
	baseTypes?: Record<string, TypeDesc>;
	children: BlockNode[];
	getApi: (providerId: string) => string;
}): AnalyzeLevelResult {
	const typesFromParams: Record<string, TypeDesc> = { ...(opts.baseTypes ?? {}) };
	for (const p of opts.existingParams) {
		if (p.userType) typesFromParams[p.name] = p.userType;
	}

	const knownScripts: Record<string, string> = {};
	for (const p of opts.existingParams) {
		if (p.resolution.kind === 'static' && p.resolution.value.trim()) {
			knownScripts[p.name] = p.resolution.value;
		}
	}

	const allScripts: ScriptEntry[] = [...opts.scripts];
	for (const p of opts.existingParams) {
		if (p.resolution.kind === 'static' && p.resolution.value.trim()) {
			allScripts.push({ source: p.resolution.value, mode: 'script' });
		}
	}

	// Phase 1: Pure type discovery (no known values).
	// All branches are live → all context refs discovered with correct types.
	const collectResult = collectUnresolvedParams({
		scripts: allScripts,
		nodeNames: opts.nodeNames,
		providedKeys: opts.providedKeys,
		contextTypes: typesFromParams,
	});

	if (!collectResult.ok) {
		return { ok: false, errors: collectResult.errors, phase: 'analysis' };
	}

	// Build full type set from Phase 1 discovery.
	const injectedTypes: Record<string, TypeDesc> = { ...typesFromParams };
	for (const k of collectResult.keys) {
		if (!injectedTypes[k.name]) {
			injectedTypes[k.name] = k.type;
		}
	}

	// Phase 2: Typecheck + Pruning.
	// Typecheck uses the full type set so all branches compile.
	const env = computeExternalContextEnv(opts.children, injectedTypes, opts.getApi);

	// Pruning: re-analyze with known values + full types to find dead branches.
	const hasKnown = Object.keys(knownScripts).length > 0;
	let unresolvedKeys = collectResult.keys;

	if (hasKnown) {
		const pruneResult = collectUnresolvedParams({
			scripts: allScripts,
			nodeNames: opts.nodeNames,
			providedKeys: opts.providedKeys,
			contextTypes: injectedTypes,
			knownScripts,
		});
		if (pruneResult.ok) {
			const prunedNames = new Set(
				pruneResult.keys.filter((k) => k.status === 'pruned').map((k) => k.name)
			);
			unresolvedKeys = collectResult.keys.filter((k) => !prunedNames.has(k.name));
		}
	}

	return { ok: true, discoveredTypes: env.contextTypes, unresolvedKeys };
}

/**
 * Merge discovered unresolved keys into existing contextParams,
 * preserving user-set resolution and userType overrides.
 * Params no longer discovered are kept with `active: false`.
 */
export function mergeDiscoveredParams(
	existing: ContextParam[],
	discovered: ContextKeyInfo[],
): ContextParam[] {
	const discoveredNames = new Set(discovered.map((k) => k.name));
	const map = new Map(existing.map((p) => [p.name, p]));

	// Active params: currently discovered
	const active: ContextParam[] = discovered.map((k) => {
		const prev = map.get(k.name);
		return {
			name: k.name,
			inferredType: k.type,
			resolution: prev?.resolution ?? { kind: 'unresolved' as const },
			userType: prev?.userType,
			editorMode: prev?.editorMode,
			active: true,
		};
	});

	// Params no longer discovered: deactivate (hidden by dead branch).
	// Configuration (resolution, userType, editorMode) is preserved —
	// if the param is rediscovered later, it reappears with saved settings.
	const inactive: ContextParam[] = existing
		.filter((p) => !discoveredNames.has(p.name))
		.map((p) => ({
			...p,
			active: false,
		}));

	return [...active, ...inactive];
}

/**
 * Build injected context types from contextParams.
 * Only uses userType (explicitly set by the user).
 * inferredType is intentionally excluded — it can be stale from a previous
 * analysis and conflict with changed templates (e.g. Int vs Enum pattern).
 * Fresh types are discovered each analysis cycle via collectUnresolvedParams.
 */
export function buildInjectedTypes(contextParams: ContextParam[]): Record<string, TypeDesc> {
	const types: Record<string, TypeDesc> = {};
	for (const p of contextParams) {
		if (p.userType) types[p.name] = p.userType;
	}
	return types;
}

/**
 * Convert a UI Node to the WebNode format expected by the WASM engine.
 * Message block references are resolved to inline templates.
 */
export function toWebNode(node: Node, api: string): WebNode {
	return {
		name: node.name,
		kind: node.kind,
		api,
		model: node.model,
		temperature: node.temperature,
		maxTokens: node.maxTokens,
		selfSpec: node.selfSpec,
		strategy: node.strategy,
		retry: node.retry,
		assert: node.assert,
		messages: node.messages.map((m) => {
			if (m.kind === 'block') {
				const template = m.source.type === 'inline' ? m.source.template : '';
				return { kind: 'block' as const, role: m.role, template };
			}
			return {
				kind: 'iterator' as const,
				iterator: m.iterator,
				role: m.role,
				slice: m.slice,
				tokenBudget: m.tokenBudget,
			};
		}),
		tools: node.tools.map((t) => ({
			name: t.name,
			description: t.description,
			node: t.nodeId,
			params: t.params,
		})),
	};
}

export type ContextEnvResult = {
	contextTypes: Record<string, TypeDesc>;
	nodeLocals: Record<string, { raw: TypeDesc; self: TypeDesc }>;
	nodeErrors: Record<string, Record<string, string>>;
};

/**
 * Compute all externally-visible context types, per-node local types,
 * and per-node per-field typecheck errors from nodes in a block tree
 * via WASM orchestration pipeline.
 */
export function computeExternalContextEnv(
	children: BlockNode[],
	injectedTypes: Record<string, TypeDesc>,
	getApi: (providerId: string) => string,
): ContextEnvResult {
	const nodes = collectNodes(children);
	const webNodes = nodes
		.filter((n) => n.name)
		.map((n) => toWebNode(n, getApi(n.providerId)));
	const result = typecheckNodes(webNodes, injectedTypes);
	if ('error' in result) {
		console.warn('[computeExternalContextEnv] error:', result.error);
		return { contextTypes: injectedTypes, nodeLocals: {}, nodeErrors: {} };
	}
	return result;
}
