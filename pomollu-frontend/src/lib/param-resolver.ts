import type { ContextKeyInfo, WebNode, TypecheckNodesResult } from './engine.js';
import type { TypeDesc } from './type-parser.js';
import { isUnknownType } from './type-parser.js';
import type { Node, BlockNode, ContextBinding, DisplayEntry, DisplayRegion, ContextParam } from './types.js';
import { isRawBlock, isScriptBlock, BUILTIN_CONTEXT_REFS } from './types.js';
import { collectBlocks, collectNodes } from './block-tree.js';
import { analyzeWithTypes, typecheckNodes } from './engine.js';

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

export function collectUnresolvedParams(opts: {
	scripts: { source: string; mode: 'script' | 'template' }[];
	nodeNames: Set<string>;
	providedKeys: Set<string>;
	contextTypes: Record<string, TypeDesc>;
}): ContextKeyInfo[] {
	const seen = new Map<string, TypeDesc>();

	for (const { source, mode } of opts.scripts) {
		if (!source.trim()) continue;
		const result = analyzeWithTypes(source, mode, opts.contextTypes);
		if (!result.ok) continue;
		for (const key of result.context_keys) {
			if (BUILTIN_CONTEXT_REFS.has(key.name)) continue;
			if (opts.nodeNames.has(key.name)) continue;
			if (opts.providedKeys.has(key.name)) continue;
			const existing = seen.get(key.name);
			if (!existing || isUnknownType(existing)) {
				seen.set(key.name, key.type);
			}
		}
	}

	return Array.from(seen.entries())
		.map(([name, type]) => ({ name, type }))
		.sort((a, b) => a.name.localeCompare(b.name));
}

/**
 * Shared analysis pipeline for Prompt/Profile/Bot settings.
 * Discovers unresolved context refs, builds injected types,
 * and runs orchestration typecheck.
 *
 * Each caller assembles the appropriate scripts/nodeNames/providedKeys
 * for its level, then calls this to get the final discoveredTypes.
 */
export function analyzeLevel(opts: {
	scripts: ScriptEntry[];
	nodeNames: Set<string>;
	providedKeys: Set<string>;
	existingParams: ContextParam[];
	baseTypes?: Record<string, TypeDesc>;
	children: BlockNode[];
	getApi: (providerId: string) => string;
}): { discoveredTypes: Record<string, TypeDesc>; unresolvedKeys: ContextKeyInfo[] } {
	const typesFromParams: Record<string, TypeDesc> = { ...(opts.baseTypes ?? {}) };
	for (const p of opts.existingParams) {
		if (p.userType) typesFromParams[p.name] = p.userType;
	}

	const keys = collectUnresolvedParams({
		scripts: opts.scripts,
		nodeNames: opts.nodeNames,
		providedKeys: opts.providedKeys,
		contextTypes: typesFromParams,
	});

	const injectedTypes: Record<string, TypeDesc> = { ...typesFromParams };
	for (const k of keys) {
		if (!isUnknownType(k.type) && !injectedTypes[k.name]) {
			injectedTypes[k.name] = k.type;
		}
	}

	const env = computeExternalContextEnv(opts.children, injectedTypes, opts.getApi);
	return { discoveredTypes: env.contextTypes, unresolvedKeys: keys };
}

/**
 * Merge discovered unresolved keys into existing contextParams,
 * preserving user-set resolution and userType overrides.
 */
export function mergeDiscoveredParams(
	existing: ContextParam[],
	discovered: ContextKeyInfo[],
): ContextParam[] {
	const map = new Map(existing.map((p) => [p.name, p]));
	return discovered.map((k) => {
		const prev = map.get(k.name);
		return {
			name: k.name,
			inferredType: k.type,
			resolution: prev?.resolution ?? { kind: 'unresolved' as const },
			userType: prev?.userType,
		} satisfies ContextParam;
	});
}

/**
 * Build injected context types from contextParams.
 * Uses userType if set, otherwise inferredType (skipping '?').
 */
export function buildInjectedTypes(contextParams: ContextParam[]): Record<string, TypeDesc> {
	const types: Record<string, TypeDesc> = {};
	for (const p of contextParams) {
		const ty = p.userType || (isUnknownType(p.inferredType) ? undefined : p.inferredType);
		if (ty) types[p.name] = ty;
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
