/**
 * AnalysisOrchestrator — 3-level hierarchy analysis using LSP DocumentManager.
 *
 * Manages Prompt → Profile → Bot topology with incremental document sync.
 * Per-document scope is constructed here (pomollu-specific).
 * DocumentManager handles pure document lifecycle.
 */

import type { ContextKeyInfo, WebNode, NodeErrors, LanguageSession, TypecheckNodesResult, CompletionItem, DocScope } from './engine.js';
import type { TypeDesc } from './type-parser.js';
import { isUnknownType } from './type-parser.js';
import type { Node, BlockNode, ContextBinding, DisplayEntry, DisplayRegion, ContextParam, ParamOverride, Prompt, Profile, Bot, ApiKind } from './types.js';
import { isRawBlock, CONTEXT_TYPE, HISTORY_ENTRY_TYPE, HISTORY_BINDING_NAME, type RawBlock } from './types.js';
import { collectBlocks, collectNodes } from './block-tree.js';
import { DocumentManager, type ScriptEntry, type CollectedKeys } from './document-manager.svelte.js';

/**
 * Engine-internal context refs — provided by the engine at runtime,
 * but type unknown at discovery time (resolved after rebuildNodes).
 *
 * These are NOT user params. Excluded from context_keys results
 * because they go into the "provided" tier of the scope after Step 5.
 * At discovery time, their types aren't known yet, so they can't be
 * in the scope — instead they're filtered via excludeSet.
 *
 * After rebuildNodes (Step 5), they ARE in the provided scope with
 * correct types, so diagnostics/completions work properly.
 */
const ENGINE_INTERNAL_REFS = new Set(['self', 'raw', 'item', 'index', 'content']);

// ---------------------------------------------------------------------------
// AnalysisOrchestrator
// ---------------------------------------------------------------------------

export type LevelResult = {
	env: ContextEnvResult;
	params: ContextParam[];
	activeParams: ContextParam[];
	ownParams: ContextParam[];
};

type GetApi = (providerId: string) => ApiKind | undefined;

export class AnalysisOrchestrator {
	private session: LanguageSession;
	private promptDocs: DocumentManager;
	private profileDocs: DocumentManager;
	private botDocs: DocumentManager;
	private fnParamDocs: DocumentManager;

	constructor(session: LanguageSession) {
		this.session = session;
		this.promptDocs = new DocumentManager(session);
		this.profileDocs = new DocumentManager(session);
		this.botDocs = new DocumentManager(session);
		this.fnParamDocs = new DocumentManager(session);
	}

	get docs(): { prompt: DocumentManager; profile: DocumentManager; bot: DocumentManager } {
		return { prompt: this.promptDocs, profile: this.profileDocs, bot: this.botDocs };
	}

	// -----------------------------------------------------------------------
	// Public API — level analysis
	// -----------------------------------------------------------------------

	analyzePrompt(prompt: Prompt, getApi: GetApi): LevelResult {
		const nodeNames = collectNodeNames(prompt.children);
		nodeNames.add('context');

		// Ensure @history always exists (forced Iterator type)
		const bindings = [...prompt.contextBindings];
		if (!bindings.some(b => b.name === HISTORY_BINDING_NAME)) {
			bindings.push({ name: HISTORY_BINDING_NAME, script: '' });
		}

		const providedKeys = new Set(bindings.map((b) => b.name).filter((n) => n));
		const provided: Record<string, TypeDesc> = { context: CONTEXT_TYPE };
		const userTypes: Record<string, TypeDesc> = {};
		applyUserTypeOverrides(userTypes, prompt.paramOverrides);

		return this.analyzeLevel(this.promptDocs, 'prompt', {
			children: prompt.children,
			bindings,
			paramOverrides: prompt.paramOverrides,
			nodeNames,
			providedKeys,
			provided,
			userTypes,
			getApi,
		});
	}

	analyzeProfile(profile: Profile, getApi: GetApi): LevelResult {
		const nodeNames = collectNodeNames(profile.children);
		nodeNames.add('context');
		const provided: Record<string, TypeDesc> = { context: CONTEXT_TYPE };
		const userTypes: Record<string, TypeDesc> = {};
		applyUserTypeOverrides(userTypes, profile.paramOverrides);

		return this.analyzeLevel(this.profileDocs, 'profile', {
			children: profile.children,
			bindings: [],
			paramOverrides: profile.paramOverrides,
			nodeNames,
			providedKeys: new Set(),
			provided,
			userTypes,
			getApi,
		});
	}

	analyzeBot(bot: Bot, prompt: Prompt, profile: Profile, getApi: GetApi): LevelResult {
		// Topology: Prompt → Profile → Bot
		const promptResult = this.analyzePrompt(prompt, getApi);
		const profileResult = this.analyzeProfile(profile, getApi);

		// provided: CONTEXT_TYPE + prompt/profile params (engine-resolved at this level)
		const provided: Record<string, TypeDesc> = { context: CONTEXT_TYPE };
		for (const p of promptResult.params) {
			provided[p.name] = p.userType ?? p.inferredType;
		}
		for (const p of profileResult.params) {
			provided[p.name] = p.userType ?? p.inferredType;
		}

		// user: bot-level param overrides only
		const userTypes: Record<string, TypeDesc> = {};
		applyUserTypeOverrides(userTypes, bot.paramOverrides);

		// providedKeys: bindings + prompt params + profile params
		const providedKeys = new Set<string>();
		for (const b of prompt.contextBindings) if (b.name) providedKeys.add(b.name);
		for (const p of promptResult.params) providedKeys.add(p.name);
		for (const p of profileResult.params) providedKeys.add(p.name);

		// nodeNames: all 3 levels
		const nodeNames = new Set<string>();
		for (const n of collectNodeNames(prompt.children)) nodeNames.add(n);
		for (const n of collectNodeNames(profile.children)) nodeNames.add(n);
		for (const n of collectNodeNames(bot.children)) nodeNames.add(n);
		nodeNames.add('context');

		// Bot-level analysis includes all 3 levels
		const allChildren = [...prompt.children, ...profile.children, ...bot.children];

		const result = this.analyzeLevel(this.botDocs, 'bot', {
			children: allChildren,
			bindings: prompt.contextBindings,
			paramOverrides: bot.paramOverrides,
			nodeNames,
			providedKeys,
			provided,
			userTypes,
			getApi,
			extraScripts: collectBotDisplayScripts(bot),
		});

		// Merge all 3 levels' params
		const ownParams = result.params;
		const allParams = [...promptResult.params, ...profileResult.params, ...ownParams];
		const allActiveParams = allParams.filter((p) => p.active !== false);

		return { env: result.env, params: allParams, activeParams: allActiveParams, ownParams };
	}

	dispose(): void {
		this.promptDocs.dispose();
		this.profileDocs.dispose();
		this.botDocs.dispose();
		this.fnParamDocs.dispose();
	}

	// -----------------------------------------------------------------------
	// Core: analyzeLevel — 6-step incremental analysis
	// -----------------------------------------------------------------------

	private analyzeLevel(
		manager: DocumentManager,
		level: string,
		opts: {
			children: BlockNode[];
			bindings: ContextBinding[];
			paramOverrides: Record<string, ParamOverride>;
			nodeNames: Set<string>;
			providedKeys: Set<string>;
			provided: Record<string, TypeDesc>;
			userTypes: Record<string, TypeDesc>;
			getApi: GetApi;
			extraScripts?: Map<string, { source: string; mode: 'script' | 'template' }>;
		},
	): LevelResult {
		const { paramOverrides, nodeNames, providedKeys, getApi } = opts;

		// Step 0: fn_param discovery (function nodes only)
		const initialScope: DocScope = { provided: opts.provided, user: opts.userTypes };
		const nodeFnParams = this.discoverFnParams(level, opts.children, initialScope, nodeNames);

		// Step 1: rebuildNodes FIRST — establishes node types + system types.
		// Uses user overrides only (no discovered params yet).
		// After this, node_env has system tier with @turn_index + node names + types.
		const flatInitial: Record<string, TypeDesc> = { ...opts.provided, ...opts.userTypes };
		const blockLookup = new Map<string, RawBlock>();
		for (const b of collectBlocks(opts.children)) {
			if (isRawBlock(b)) blockLookup.set(b.id, b);
		}
		const allNodes = collectNodes(opts.children).filter((n) => n.name);
		const webNodes: import('./engine.js').WebNode[] = allNodes
			.map((n) => toWebNode(n, getApi(n.providerId), nodeFnParams[n.name], blockLookup, allNodes));
		// Context bindings → Expr nodes
		for (const b of opts.bindings) {
			if (!b.name) continue;
			// history is always registered (forced Iterator type), others need script
			if (b.name === HISTORY_BINDING_NAME || b.script.trim()) {
				webNodes.push(bindingToExprNode(b.name, b.script));
			}
		}
		const typecheckResult = this.session.rebuildNodes(webNodes, flatInitial);

		const EMPTY_ENV: ContextEnvResult = { contextTypes: {}, nodeLocals: {}, nodeErrors: {}, nodeFnParams: {}, scriptErrors: {} };
		const env: ContextEnvResult = typecheckResult.envErrors.length > 0
			? EMPTY_ENV
			: { ...typecheckResult, nodeFnParams, scriptErrors: {} };

		// Step 2: Build provided scope from rebuildNodes contextTypes.
		// contextTypes includes system tier (turn_index, node names with correct types).
		// These go into provided tier → context_keys excludes them.
		const provided: Record<string, TypeDesc> = { ...opts.provided };
		if (typecheckResult.envErrors.length === 0) {
			for (const [name, ty] of Object.entries(typecheckResult.contextTypes)) {
				// contextTypes = system + user merged.
				// Only add names that are NOT user-declared params.
				if (!opts.userTypes[name]) {
					provided[name] = ty;
				}
			}
		}

		// Step 3: Sync documents with scope (provided has real node types now)
		const baseDocScope: DocScope = { provided, user: opts.userTypes };
		const scripts = buildKeyedScripts(level, opts.children, opts.bindings, paramOverrides, baseDocScope);
		if (opts.extraScripts) {
			for (const [key, entry] of opts.extraScripts) {
				scripts.set(key, { source: entry.source, mode: entry.mode, scope: baseDocScope });
			}
		}
		manager.sync(scripts);

		// Step 3.5: Bind node documents to inference units
		for (const [key] of scripts) {
			const parts = key.split(':');
			// Bindings are Expr WebNodes — bind them too
			if (parts.length >= 3 && parts[1] === 'binding') {
				const bindingName = parts[2];
				manager.bindDocToNode(key, bindingName, 'exprSource');
			}
			if (parts.length >= 4 && parts[1] === 'node') {
				const nodeName = parts[2];
				const field = parts[3];
				const fieldMap: Record<string, [string, number | undefined]> = {
					'init': ['initialValue', undefined],
					'bind': ['bind', undefined],
					'assert': ['assert', undefined],
					'ifmod': ['ifModifiedKey', undefined],
					'expr': ['exprSource', undefined],
				};
				if (fieldMap[field]) {
					const [apiField, idx] = fieldMap[field];
					manager.bindDocToNode(key, nodeName, apiField, idx);
				} else if (field === 'msg' && parts[4] !== undefined) {
					const msgIdx = parseInt(parts[4]);
					if (parts[5] === 'tmpl') {
						manager.bindDocToNode(key, nodeName, 'iteratorTmpl', msgIdx);
					} else if (parts[5] === 'iter') {
						manager.bindDocToNode(key, nodeName, 'iteratorExpr', msgIdx);
					} else {
						manager.bindDocToNode(key, nodeName, 'message', msgIdx);
					}
				}
			}
		}

		// Step 4: Discovery — collectAllKeys (no known)
		// ENGINE_INTERNAL_REFS: @self/@raw/@item/@index — types depend on per-node context,
		// added to per-document scope in Step 6 after nodeLocals are known.
		const excludeSet = new Set([...providedKeys, ...ENGINE_INTERNAL_REFS]);
		const discovery = manager.collectAllKeys({ exclude: excludeSet });

		// Step 5: Scope Enrich — add discovered param types to user tier
		const enrichedUser = { ...opts.userTypes };
		for (const k of discovery.keys) {
			if (!enrichedUser[k.name]) enrichedUser[k.name] = k.type;
		}

		// Step 6: 2nd rebuild — full typecheck with discovered params.
		// 1st rebuild established system types (turn_index, node names).
		// 2nd rebuild includes discovered params → accurate nodeErrors + nodeLocals.
		// IMPORTANT: only pass CONTEXT_TYPE + user params. Do NOT pass system types
		// (turn_index, node names) — compute_external_context_env adds those to system tier.
		// Passing them as user would conflict with system tier.
		const flatFull: Record<string, TypeDesc> = { ...opts.provided, ...enrichedUser };
		const fullTypecheckResult = this.session.rebuildNodes(webNodes, flatFull);
		const fullEnv: ContextEnvResult = fullTypecheckResult.envErrors.length > 0
			? EMPTY_ENV
			: { ...fullTypecheckResult, nodeFnParams };

		// Step 7: Scope Finalize — add nodeLocals (@self/@raw) to provided tier
		const enrichedDocScope: DocScope = { provided, user: enrichedUser };
		if (fullTypecheckResult.envErrors.length === 0) {
			for (const [key] of scripts) {
				const docScope = buildDocScope(key, enrichedDocScope, fullTypecheckResult);
				manager.updateScope(key, docScope);
			}
		}

		// Step 8: Notify DocumentManager that rebuild completed → recompute diagnostics.
		// Bound documents get inference results (shared subst, accurate).
		// Unbound documents get standalone check_script.
		manager.onRebuildComplete();

		// Step 9: Pruning — re-query with known values
		const knownScripts: Record<string, string> = {};
		for (const [name, ov] of Object.entries(paramOverrides)) {
			if (ov.resolution.kind === 'static' && ov.resolution.value.trim()) {
				knownScripts[name] = ov.resolution.value;
			}
		}

		const discoveryNames = new Set(discovery.keys.map((k) => k.name));
		const survivingNames = new Set(discoveryNames);

		if (Object.keys(knownScripts).length > 0) {
			const pruningResult = manager.collectAllKeys({ exclude: excludeSet }, knownScripts);
			for (const k of pruningResult.keys) {
				if (k.status === 'pruned') survivingNames.delete(k.name);
			}
		}

		// Step 10: Sanitization
		const params = sanitizeParams(discovery, paramOverrides, survivingNames);
		const activeParams = params.filter((p) => p.active);

		return { env: { ...fullEnv, scriptErrors: {} }, params, activeParams, ownParams: params };
	}

	// nodeErrorsForKey removed — inference results are now stored in LspSession
	// and returned automatically by diagnostics(docId) for bound documents.

	// -----------------------------------------------------------------------
	// fn_param discovery
	// -----------------------------------------------------------------------

	private discoverFnParams(
		level: string,
		children: BlockNode[],
		baseScope: DocScope,
		nodeNames: Set<string>,
	): Record<string, DiscoveredFnParam[]> {
		const result: Record<string, DiscoveredFnParam[]> = {};
		const scripts = new Map<string, ScriptEntry>();

		for (const node of collectNodes(children)) {
			if (!node.isFunction || !node.name) continue;
			const nodeScripts: { source: string; mode: 'script' | 'template' }[] = [];
			collectScriptsFromNode(node, nodeScripts);
			if (nodeScripts.length === 0) continue;

			// Register function node scripts as documents
			for (let i = 0; i < nodeScripts.length; i++) {
				const s = nodeScripts[i];
				if (!s.source.trim()) continue;
				scripts.set(`${level}:fn:${node.name}:${i}`, {
					source: s.source,
					mode: s.mode,
					scope: baseScope,
				});
			}
		}

		this.fnParamDocs.sync(scripts);

		// Collect keys per function node
		for (const node of collectNodes(children)) {
			if (!node.isFunction || !node.name) continue;
			const seen = new Map<string, TypeDesc>();

			// Query each of this node's documents
			for (let i = 0; ; i++) {
				const key = `${level}:fn:${node.name}:${i}`;
				if (!this.fnParamDocs.has(key)) break;
				const keys = this.fnParamDocs.contextKeys(key);
				for (const k of keys) {
					if (nodeNames.has(k.name)) continue;
					if (!seen.has(k.name) || isUnknownType(seen.get(k.name)!)) {
						seen.set(k.name, k.type);
					}
				}
			}

			if (seen.size > 0) {
				result[node.name] = Array.from(seen.entries())
					.map(([name, inferredType]) => ({ name, inferredType }))
					.sort((a, b) => a.name.localeCompare(b.name));
			}
		}

		return result;
	}
}

// ---------------------------------------------------------------------------
// Script collection helpers
// ---------------------------------------------------------------------------

function collectScriptsFromNode(node: Node, out: { source: string; mode: 'script' | 'template' }[]) {
	if (node.strategy.initialValue?.trim()) out.push({ source: node.strategy.initialValue, mode: 'script' });
	if (node.strategy.assert?.trim() && node.strategy.assert !== 'true') out.push({ source: node.strategy.assert, mode: 'script' });
	if (node.strategy.execution?.mode === 'if-modified' && node.strategy.execution.key?.trim()) {
		out.push({ source: node.strategy.execution.key, mode: 'script' });
	}
	for (const msg of node.messages) {
		if (msg.kind === 'block' && msg.source.type === 'inline') {
			out.push({ source: msg.source.template, mode: 'template' });
		}
		if (msg.kind === 'iterator' && msg.iterator.trim()) {
			out.push({ source: msg.iterator, mode: 'script' });
		}
	}
	if (node.kind === 'expr' && node.exprSource.trim()) {
		out.push({ source: node.exprSource, mode: 'script' });
	}
	if ((node.strategy.persistency?.kind === 'sequence' || node.strategy.persistency?.kind === 'patch') && node.strategy.persistency.bind?.trim()) {
		out.push({ source: node.strategy.persistency.bind, mode: 'script' });
	}
}

/**
 * Build keyed scripts map with per-document scope.
 *
 * Each script gets a unique key ({level}:{kind}:{id}:{field})
 * and the appropriate scope based on its position.
 */
function buildKeyedScripts(
	level: string,
	children: BlockNode[],
	bindings: ContextBinding[],
	paramOverrides: Record<string, ParamOverride>,
	baseScope: DocScope,
): Map<string, ScriptEntry> {
	const scripts = new Map<string, ScriptEntry>();

	// Bindings
	for (const b of bindings) {
		if (!b.script.trim()) continue;
		scripts.set(`${level}:binding:${b.name}`, {
			source: b.script,
			mode: 'script',
			scope: baseScope,
		});
	}

	// Raw blocks
	for (const block of collectBlocks(children)) {
		if (isRawBlock(block) && block.text.trim()) {
			scripts.set(`${level}:rawblock:${block.id}`, {
				source: block.text,
				mode: block.mode,
				scope: baseScope,
			});
		}
	}

	// Node fields (skip function nodes — handled by fn_param discovery)
	for (const node of collectNodes(children)) {
		if (node.isFunction || !node.name) continue;
		addNodeScripts(scripts, level, node, baseScope);
	}

	// Param static value scripts
	for (const [name, ov] of Object.entries(paramOverrides)) {
		if (ov.resolution.kind === 'static' && ov.resolution.value.trim()) {
			scripts.set(`${level}:param:${name}:static`, {
				source: ov.resolution.value,
				mode: 'script',
				scope: baseScope,
			});
		}
	}

	return scripts;
}

function addNodeScripts(
	scripts: Map<string, ScriptEntry>,
	level: string,
	node: Node,
	baseScope: DocScope,
): void {
	const prefix = `${level}:node:${node.name}`;

	// All node scripts use baseScope initially.
	// Step 5 (buildDocScope) adds @self/@raw to provided tier after rebuildNodes.

	if (node.kind === 'expr' && node.exprSource.trim()) {
		scripts.set(`${prefix}:expr`, {
			source: node.exprSource,
			mode: 'script',
			scope: baseScope,
		});
	}

	if (node.strategy.initialValue?.trim()) {
		scripts.set(`${prefix}:init`, {
			source: node.strategy.initialValue,
			mode: 'script',
			scope: baseScope,
		});
	}

	if ((node.strategy.persistency?.kind === 'sequence' || node.strategy.persistency?.kind === 'patch') && node.strategy.persistency.bind?.trim()) {
		scripts.set(`${prefix}:bind`, {
			source: node.strategy.persistency.bind,
			mode: 'script',
			scope: baseScope,
		});
	}

	if (node.strategy.assert?.trim() && node.strategy.assert !== 'true') {
		scripts.set(`${prefix}:assert`, {
			source: node.strategy.assert,
			mode: 'script',
			scope: baseScope,
		});
	}

	if (node.strategy.execution?.mode === 'if-modified' && node.strategy.execution.key?.trim()) {
		scripts.set(`${prefix}:ifmod`, {
			source: node.strategy.execution.key,
			mode: 'script',
			scope: baseScope,
		});
	}

	for (let i = 0; i < node.messages.length; i++) {
		const msg = node.messages[i];
		if (msg.kind === 'block' && msg.source.type === 'inline' && msg.source.template.trim()) {
			scripts.set(`${prefix}:msg:${i}`, {
				source: msg.source.template,
				mode: 'template',
				scope: baseScope,
			});
		}
		if (msg.kind === 'iterator' && msg.iterator.trim()) {
			scripts.set(`${prefix}:msg:${i}:iter`, {
				source: msg.iterator,
				mode: 'script',
				scope: baseScope,
			});
		}
		// Iterator template: @item + @index in provided tier (engine loop variables)
		if (msg.kind === 'iterator' && msg.template?.trim()) {
			const iterScope: DocScope = {
				provided: {
					...baseScope.provided,
					item: { kind: 'unsupported', raw: '?' },
					index: { kind: 'primitive', name: 'int' },
				},
				user: baseScope.user,
			};
			scripts.set(`${prefix}:msg:${i}:tmpl`, {
				source: msg.template,
				mode: 'template',
				scope: iterScope,
			});
		}
	}
}

function collectBotDisplayScripts(bot: Bot): Map<string, { source: string; mode: 'script' | 'template' }> {
	const scripts = new Map<string, { source: string; mode: 'script' | 'template' }>();

	if (bot.display.iterator.trim()) {
		scripts.set('bot:display:iterator', { source: bot.display.iterator, mode: 'script' });
	}

	for (const region of bot.regions) {
		if (region.kind === 'static' && region.template.trim()) {
			scripts.set(`bot:region:${region.id}:template`, { source: region.template, mode: 'template' });
		} else if (region.kind === 'iterable') {
			if (region.iterator.trim()) {
				scripts.set(`bot:region:${region.id}:iterator`, { source: region.iterator, mode: 'script' });
			}
			for (const entry of region.entries) {
				if (entry.condition.trim()) {
					scripts.set(`bot:region:${region.id}:entry:${entry.id}:cond`, { source: entry.condition, mode: 'script' });
				}
				if (entry.template.trim()) {
					scripts.set(`bot:region:${region.id}:entry:${entry.id}:tmpl`, { source: entry.template, mode: 'template' });
				}
			}
		}
	}

	return scripts;
}

export function collectNodeNames(children: BlockNode[]): Set<string> {
	return new Set(collectNodes(children).map((n) => n.name).filter((n) => n));
}

// ---------------------------------------------------------------------------
// Per-document scope construction
// ---------------------------------------------------------------------------

/**
 * Build the appropriate DocScope for a document based on its key.
 *
 * @self/@raw go into the **provided** tier (engine-internal, not user params).
 * After rebuildNodes (typecheckResult available), exact types are used.
 * Before rebuildNodes (typecheckResult undefined), no @self/@raw in scope.
 */
function buildDocScope(
	key: string,
	baseScope: DocScope,
	typecheckResult: TypecheckNodesResult | undefined,
): DocScope {
	const parts = key.split(':');
	// {level}:node:{name}:{field}
	if (parts.length >= 4 && parts[1] === 'node') {
		const nodeName = parts[2];
		const field = parts[3];
		const locals = typecheckResult?.nodeLocals[nodeName];

		// Check for iterator template: {level}:node:{name}:msg:{i}:tmpl
		const isIterTemplate = parts.length >= 6 && parts[5] === 'tmpl';

		if (isIterTemplate) {
			// Iterator template scope: @self + @item + @index in provided tier
			const iterProvided: Record<string, TypeDesc> = { ...baseScope.provided };
			if (locals) iterProvided['self'] = locals.self;
			iterProvided['item'] = { kind: 'unsupported', raw: '?' }; // TODO: refine from iterator expr type
			iterProvided['index'] = { kind: 'primitive', name: 'int' };
			return { provided: iterProvided, user: baseScope.user };
		}

		if (locals) {
			switch (field) {
				case 'expr':
				case 'msg': {
					// Body scope: @self in provided tier
					return { provided: { ...baseScope.provided, self: locals.self }, user: baseScope.user };
				}
				case 'bind':
				case 'assert': {
					// Bind scope: @self + @raw in provided tier
					return { provided: { ...baseScope.provided, self: locals.self, raw: locals.raw }, user: baseScope.user };
				}
				case 'init':
				case 'ifmod': {
					// InitialValue scope: no @self, no @raw
					return baseScope;
				}
			}
		}
	}

	return baseScope;
}

// ---------------------------------------------------------------------------
// Sanitization
// ---------------------------------------------------------------------------

function sanitizeParams(
	discovery: CollectedKeys,
	paramOverrides: Record<string, ParamOverride>,
	survivingNames: Set<string>,
): ContextParam[] {
	const params: ContextParam[] = discovery.keys.map((k) => {
		const ov = paramOverrides[k.name];
		return {
			name: k.name,
			inferredType: k.type,
			resolution: ov?.resolution ?? { kind: 'unresolved' as const },
			userType: ov?.userType,
			editorMode: ov?.editorMode,
			active: survivingNames.has(k.name),
		};
	});

	// Preserve overrides for scripts that failed (hasSkippedScripts)
	if (discovery.hasSkippedScripts) {
		const discoveredNames = new Set(discovery.keys.map((k) => k.name));
		for (const [name, ov] of Object.entries(paramOverrides)) {
			if (!discoveredNames.has(name)) {
				params.push({
					name,
					inferredType: { kind: 'unsupported', raw: '?' },
					resolution: ov.resolution,
					userType: ov.userType,
					editorMode: ov.editorMode,
				});
			}
		}
		params.sort((a, b) => a.name.localeCompare(b.name));
	}

	return params;
}

function applyUserTypeOverrides(scope: Record<string, TypeDesc>, overrides: Record<string, ParamOverride>): void {
	for (const [name, ov] of Object.entries(overrides)) {
		if (ov.userType) scope[name] = ov.userType;
	}
}

// ---------------------------------------------------------------------------
// Pure helpers (kept from original)
// ---------------------------------------------------------------------------

export function mergeParams(
	analysisParams: ContextParam[],
	overrides: Record<string, ParamOverride>
): ContextParam[] {
	return analysisParams.map(p => {
		const ov = overrides[p.name];
		if (!ov) return p;
		return { ...p, resolution: ov.resolution, userType: ov.userType, editorMode: ov.editorMode };
	});
}

export function pruneOverrides(
	overrides: Record<string, ParamOverride>,
	analysisParams: ContextParam[]
): Record<string, ParamOverride> | null {
	const liveNames = new Set(analysisParams.map(p => p.name));
	const staleKeys = Object.keys(overrides).filter(k => !liveNames.has(k));
	if (staleKeys.length === 0) return null;
	const pruned = { ...overrides };
	for (const k of staleKeys) delete pruned[k];
	return pruned;
}

export function typeDescToFnParamString(desc: TypeDesc): string {
	if (desc.kind === 'primitive') return desc.name;
	return '';
}

// ---------------------------------------------------------------------------
// WebNode conversion (kept from original)
// ---------------------------------------------------------------------------

function sanitizeExecution(execution: import('./types.js').Execution | undefined): import('./types.js').Execution {
	switch (execution?.mode) {
		case 'always':
		case 'once-per-turn':
		case 'if-modified':
			return execution;
		default:
			return { mode: 'always' };
	}
}

function sanitizePersistency(persistency: import('./types.js').Persistency | undefined): import('./types.js').Persistency {
	switch (persistency?.kind) {
		case 'ephemeral':
		case 'snapshot':
		case 'sequence':
		case 'patch':
			return persistency;
		default:
			return { kind: 'ephemeral' };
	}
}

export function toWebNode(node: Node, api: ApiKind | undefined, discoveredFnParams?: DiscoveredFnParam[], blockLookup?: Map<string, RawBlock>, allNodes?: Node[]): WebNode {
	const shared = {
		name: node.name,
		strategy: {
			execution: sanitizeExecution(node.strategy.execution),
			persistency: sanitizePersistency(node.strategy.persistency),
			initialValue: node.strategy.initialValue || undefined,
			retry: node.strategy.retry ?? 0,
			assert: node.strategy.assert ?? '',
		},
		isFunction: node.isFunction ?? false,
		fnParams: discoveredFnParams
			? discoveredFnParams.map((p) => {
					const stored = (node.fnParams ?? []).find((fp) => fp.name === p.name);
					return { name: p.name, type: stored?.type || typeDescToFnParamString(p.inferredType) };
				})
			: node.fnParams ?? [],
	};

	switch (node.kind) {
		case 'llm':
			return {
				...shared,
				kind: 'llm',
				api,
				model: node.model,
				temperature: node.temperature,
				topP: node.topP ?? null,
				topK: node.topK ?? null,
				grounding: node.grounding ?? false,
				thinking: node.thinking,
				maxTokens: node.maxTokens,
				messages: node.messages.map((m) => {
					if (m.kind === 'block') {
						let template = '';
						if (m.source.type === 'inline') {
							template = m.source.template;
						} else if (blockLookup) {
							template = blockLookup.get(m.source.blockId)?.text ?? '';
						}
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
				tools: node.tools.map((t) => {
					const target = allNodes?.find((n) => n.id === t.nodeId);
					return {
						name: target?.name ?? '',
						description: t.description,
						node: target?.name ?? '',
						params: (target?.fnParams ?? []).map((p) => ({
							name: p.name,
							type: p.type,
							description: p.description,
						})),
					};
				}).filter((t) => t.node !== ''),
			};
		case 'expr':
			return {
				...shared,
				kind: 'expr',
				exprSource: node.exprSource,
			};
		case 'plain':
			return {
				...shared,
				kind: 'plain',
			};
	}
}

export type DiscoveredFnParam = { name: string; inferredType: TypeDesc };

/** Convert a context binding to an Expr WebNode.
 * `history` binding is forced to Iterator<{content, content_type, role}>.
 * Other bindings infer type from script. */
export function bindingToExprNode(name: string, script: string): WebNode {
	return {
		name,
		kind: 'expr',
		exprSource: script,
		outputTy: name === HISTORY_BINDING_NAME ? HISTORY_ENTRY_TYPE : undefined,
		strategy: {
			execution: { mode: 'once-per-turn' },
			persistency: { kind: 'ephemeral' },
			retry: 0,
			assert: '',
		},
		isFunction: false,
		fnParams: [],
	};
}

export type ContextEnvResult = {
	contextTypes: Record<string, TypeDesc>;
	nodeLocals: Record<string, { raw: TypeDesc; self: TypeDesc }>;
	nodeErrors: Record<string, NodeErrors>;
	nodeFnParams: Record<string, DiscoveredFnParam[]>;
	/** Per-document diagnostics — keyed by document key. Reactive via ownerEnv. */
	scriptErrors: Record<string, import('./engine.js').EngineError[]>;
};

// Re-export
export type { ScriptEntry } from './document-manager.svelte.js';

// Display/Region script collection — used by bot-settings and session-builder
export function collectScriptsFromDisplay(entries: DisplayEntry[]): { source: string; mode: 'script' | 'template' }[] {
	const scripts: { source: string; mode: 'script' | 'template' }[] = [];
	for (const entry of entries) {
		if (entry.condition.trim()) scripts.push({ source: entry.condition, mode: 'script' });
		if (entry.template.trim()) scripts.push({ source: entry.template, mode: 'template' });
	}
	return scripts;
}

export function collectScriptsFromRegions(regions: DisplayRegion[]): { source: string; mode: 'script' | 'template' }[] {
	const scripts: { source: string; mode: 'script' | 'template' }[] = [];
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

export function collectScriptsFromBindings(bindings: ContextBinding[]): { source: string; mode: 'script' | 'template' }[] {
	return bindings
		.filter((b) => b.script.trim())
		.map((b) => ({ source: b.script, mode: 'script' as const }));
}

export function collectScriptsFromTree(children: BlockNode[], opts?: { skipFunctionNodes?: boolean }): { source: string; mode: 'script' | 'template' }[] {
	const scripts: { source: string; mode: 'script' | 'template' }[] = [];
	for (const block of collectBlocks(children)) {
		if (isRawBlock(block)) {
			scripts.push({ source: block.text, mode: block.mode });
		}
	}
	for (const node of collectNodes(children)) {
		if (opts?.skipFunctionNodes && node.isFunction) continue;
		collectScriptsFromNode(node, scripts);
	}
	return scripts;
}
