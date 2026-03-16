import type { Bot, Prompt, Profile, Node, MessageDef, Block, RawBlock, ContextBlock, ContextBinding } from './types.js';
import { isRawBlock, isContextBlock, CONTEXT_TYPE, HISTORY_BINDING_NAME } from './types.js';
import type { SessionConfig, NodeConfig, MessageConfig, ProviderConfig, ExecutionConfig, PersistencyConfig, StrategyConfig } from './engine.js';
import { LanguageSession } from './engine.js';
import type { TypeDesc } from './type-parser.js';
import { isUnknownType } from './type-parser.js';
import { collectNodes, collectBlocks } from './block-tree.js';
import { promptStore, profileStore, providerStore } from './stores.svelte.js';
import { AnalysisOrchestrator, bindingToExprNode, typeDescToFnParamString, type DiscoveredFnParam } from './param-resolver.js';

function convertMessage(msg: MessageDef, blockLookup: Map<string, RawBlock>): MessageConfig {
	switch (msg.kind) {
		case 'block': {
			let template = '';
			if (msg.source.type === 'inline') {
				template = msg.source.template;
			} else {
				const raw = blockLookup.get(msg.source.blockId);
				template = raw?.text ?? '';
			}
			return {
				role: msg.role,
				inline_template: template
			};
		}
		case 'iterator':
			return {
				iterator: msg.iterator,
				role: msg.role,
				slice: msg.slice,
				token_budget: msg.tokenBudget
			};
	}
}

function convertStrategy(node: Node): StrategyConfig {
	return {
		execution: convertExecution(node.strategy.execution),
		persistency: convertPersistency(node.strategy.persistency),
		initial_value: node.strategy.initialValue?.trim() || undefined,
		retry: node.strategy.retry,
		assert_script: node.strategy.assert?.trim() || undefined,
	};
}

/** Resolve fn_params: merge stored (user overrides) with discovered (inferred types).
 *  Discovered params are the source of truth for which params exist;
 *  stored params provide user type overrides and descriptions. */
function resolveFnParams(node: Node, discovered: DiscoveredFnParam[]): { name: string; type: string; description?: string }[] {
	return discovered.map((dp) => {
		const stored = node.fnParams.find((fp) => fp.name === dp.name);
		return {
			name: dp.name,
			type: stored?.type || (isUnknownType(dp.inferredType) ? '' : typeDescToFnParamString(dp.inferredType)),
			description: stored?.description,
		};
	});
}

function convertNode(
	node: Node,
	blockLookup: Map<string, RawBlock>,
	allNodes: Node[],
	errors: string[],
	allDiscovered: Record<string, DiscoveredFnParam[]>,
): NodeConfig {
	// Resolve fn_params from discovered + stored for this node (and used for tool param derivation)
	const resolvedFn = node.isFunction ? resolveFnParams(node, allDiscovered[node.name] ?? []) : undefined;

	if (node.kind === 'expr') {
		return {
			name: node.name,
			kind: 'expr',
			template: node.exprSource,
			strategy: convertStrategy(node),
			is_function: node.isFunction || undefined,
			fn_params: resolvedFn,
		};
	}

	const provider = node.kind === 'llm' ? providerStore.get(node.providerId) : undefined;

	if (node.kind === 'llm' && !provider) {
		errors.push(`node '${node.name}': provider not found`);
	}

	// Resolve tool bindings — each tool references another node by id.
	// Params are derived from the target function node's discovered fn_params.
	const tools = node.tools.map((t) => {
		const targetNode = allNodes.find((n) => n.id === t.nodeId);
		if (!targetNode) {
			errors.push(`node '${node.name}': tool references unknown node`);
		}
		const targetParams = targetNode
			? resolveFnParams(targetNode, allDiscovered[targetNode.name] ?? [])
			: [];
		return {
			name: targetNode?.name ?? '',
			description: t.description,
			node: targetNode?.name ?? '',
			params: targetParams,
		};
	}).filter((t) => t.node !== '');

	return {
		name: node.name,
		kind: node.kind,
		strategy: convertStrategy(node),
		provider: provider?.name ?? '',
		api: provider?.api,
		model: node.model,
		temperature: node.temperature,
		top_p: node.topP,
		top_k: node.topK,
		grounding: node.grounding,
		thinking: node.thinking,
		max_tokens: { input: node.maxTokens.input, output: node.maxTokens.output },
		messages: node.messages.map((m) => convertMessage(m, blockLookup)),
		tools,
		is_function: node.isFunction || undefined,
		fn_params: resolvedFn,
	};
}

function convertExecution(execution: import('./types.js').Execution | undefined): ExecutionConfig {
	switch (execution?.mode) {
		case 'always':
			return { mode: 'always' };
		case 'once-per-turn':
			return { mode: 'once-per-turn' };
		case 'if-modified':
			return { mode: 'if-modified', key: execution.key };
		default:
			return { mode: 'always' };
	}
}

function convertPersistency(persistency: import('./types.js').Persistency | undefined): PersistencyConfig {
	switch (persistency?.kind) {
		case 'snapshot': return { kind: 'snapshot' };
		case 'sequence': return { kind: 'sequence', bind: persistency.bind };
		case 'patch': return { kind: 'patch', bind: persistency.bind };
		default: return { kind: 'ephemeral' };
	}
}

/** Convert a WebNode (from bindingToExprNode) to a NodeConfig for ChatSession. */
function webNodeToNodeConfig(wn: import('./engine.js').WebNode): NodeConfig {
	const strategy: StrategyConfig = {
		execution: wn.strategy.execution,
		persistency: wn.strategy.persistency,
		retry: wn.strategy.retry,
		assert_script: wn.strategy.assert || undefined,
		initial_value: wn.strategy.initialValue,
	};
	if (wn.kind === 'expr') {
		return { name: wn.name, kind: 'expr', template: wn.exprSource, output_ty: wn.outputTy, strategy };
	}
	return { name: wn.name, kind: wn.kind, strategy };
}

function escapeAcvusString(s: string): string {
	return '"' + s.replace(/\\/g, '\\\\').replace(/"/g, '\\"') + '"';
}

function tagsToAcvus(tags: Record<string, string>): string {
	const entries = Object.entries(tags).map(([k, v]) => `{key: ${escapeAcvusString(k)}, value: ${escapeAcvusString(v)},}`);
	return `[${entries.join(', ')}]`;
}

/**
 * Build `@context` — an object grouped by ContextBlock type.
 * Each text content part → template-mode Expr node (evaluates to String).
 * Main `context` node → script-mode Expr assembling the grouped structure.
 *
 * @context.{type} = List<{ name, description, tags, content, content_type }>
 */
function buildContextNodes(allBlocks: Block[], bot: Bot): NodeConfig[] {
	const nodes: NodeConfig[] = [];
	// Group entries by type: type → list of acvus object literals
	const groups = new Map<string, string[]>();
	let ctxIdx = 0;

	for (const b of allBlocks) {
		if (!isContextBlock(b)) continue;
		if (!b.enabled || b.info.type === 'disabled' || !b.name) continue;

		for (let i = 0; i < b.content.length; i++) {
			const part = b.content[i];
			if (part.content_type !== 'text') continue;

			// Each text content part → plain node (template mode → String)
			const partNodeName = `__ctx_${ctxIdx++}`;
			nodes.push({
				name: partNodeName,
				kind: 'plain',
				template: part.value,
				strategy: {
					execution: { mode: 'once-per-turn' },
					persistency: { kind: 'ephemeral' },
					retry: 0,
				},
			});

			const entry = `{name: ${escapeAcvusString(b.name)}, description: ${escapeAcvusString(b.info.description)}, tags: ${tagsToAcvus(b.info.tags)}, content: @${partNodeName}, content_type: ${escapeAcvusString(part.content_type)},}`;

			const type = b.info.type;
			if (!groups.has(type)) groups.set(type, []);
			groups.get(type)!.push(entry);
		}
	}

	// Always create @context with all standard fields (empty lists by default)
	const standardTypes = ['system', 'character', 'world_info', 'lorebook', 'memory'];
	const fields: string[] = [];
	for (const type of standardTypes) {
		const entries = groups.get(type) ?? [];
		fields.push(`${type}: [${entries.join(', ')}]`);
	}
	// Custom types go into context.custom with an extra `type` field
	const customEntries: string[] = [];
	for (const [type, entries] of groups) {
		if (!standardTypes.includes(type)) {
			customEntries.push(...entries);
		}
	}
	fields.push(`custom: [${customEntries.join(', ')}]`);
	fields.push(`bot_name: ${escapeAcvusString(bot.name)}`);

	nodes.push({
		name: 'context',
		kind: 'expr',
		template: `{${fields.join(', ')},}`,
		strategy: {
			execution: { mode: 'once-per-turn' },
			persistency: { kind: 'ephemeral' },
			retry: 0,
		},
		output_ty: CONTEXT_TYPE
	});

	return nodes;
}

export type BuildResult =
	| { ok: true; config: SessionConfig }
	| { ok: false; errors: string[] };

export function buildSessionConfig(bot: Bot): BuildResult | null {
	const prompt = promptStore.get(bot.promptId);
	if (!prompt) return { ok: false as const, errors: ['Prompt not found. Select a prompt in bot settings.'] };
	const profile = profileStore.get(bot.profileId);
	if (!profile) return { ok: false as const, errors: ['Profile not found. Select a profile in bot settings.'] };

	// Collect all blocks (for RawBlock lookup + ScriptBlock → Expr nodes)
	const allBlocks: Block[] = [];
	collectBlocks(prompt.children, allBlocks);
	collectBlocks(profile.children, allBlocks);
	collectBlocks(bot.children, allBlocks);

	const blockLookup = new Map<string, RawBlock>();
	for (const b of allBlocks) {
		if (isRawBlock(b)) blockLookup.set(b.id, b);
	}

	// Collect all user-defined nodes from prompt, profile, bot
	const allNodes: Node[] = [];
	collectNodes(prompt.children, allNodes);
	collectNodes(profile.children, allNodes);
	collectNodes(bot.children, allNodes);

	// Validate node name uniqueness
	const errors: string[] = [];
	const seenNodeNames = new Set<string>();
	for (const n of allNodes) {
		if (!n.name) continue;
		if (seenNodeNames.has(n.name)) {
			errors.push(`duplicate node name '${n.name}'`);
		}
		seenNodeNames.add(n.name);
	}

	// analyzeBot typechecks all 3 levels and returns ALL params merged
	// (prompt + profile + bot). This is the single source of truth.
	const session = LanguageSession.create();
	const orchestrator = new AnalysisOrchestrator(session);
	const analysisResult = orchestrator.analyzeBot(bot, prompt, profile, (id) => providerStore.get(id)?.api);
	orchestrator.dispose();
	session.free();
	const allDiscovered = analysisResult.env.nodeFnParams;


	// Convert user-defined nodes → NodeConfigs
	const nodeConfigs: NodeConfig[] = allNodes.map((n) => convertNode(n, blockLookup, allNodes, errors, allDiscovered));

	// Prompt contextBindings → Expr nodes (also tracked as side effects)
	const sideEffects: string[] = [];
	for (const binding of prompt.contextBindings) {
		if (binding.name && binding.script.trim()) {
			if (seenNodeNames.has(binding.name)) {
				errors.push(`binding '${binding.name}' conflicts with a node name`);
			}
			seenNodeNames.add(binding.name);
			nodeConfigs.push(webNodeToNodeConfig(bindingToExprNode(binding.name, binding.script)));
			sideEffects.push(binding.name);
		}
	}

	// ContextBlocks → @context object grouped by type
	nodeConfigs.push(...buildContextNodes(allBlocks, bot));

	if (nodeConfigs.length === 0) return null;

	// Collect unique providers
	const providers: Record<string, ProviderConfig> = {};
	for (const node of allNodes) {
		const p = providerStore.get(node.providerId);
		if (p && !providers[p.name]) {
			providers[p.name] = {
				api: p.api,
				endpoint: p.endpoint,
				api_key: p.apiKey
			};
		}
	}

	// Use ALL params for the session config, not just active ones.
	// active flag is for UI display (hiding pruned params in editor) — not for compilation.
	// Dead-branch params are harmless: WASM simply won't evaluate them at runtime.
	const allActiveParams = analysisResult.params;

	const context: Record<string, { type?: TypeDesc }> = {};
	for (const param of allActiveParams) {
		const ty: TypeDesc | undefined = param.userType || (isUnknownType(param.inferredType) ? undefined : param.inferredType);
		if (param.resolution.kind === 'static') {
			const script = (param.resolution as { kind: 'static'; value: string }).value;
			if (!script.trim()) {
				errors.push(`static param '${param.name}': empty script`);
			}
			nodeConfigs.push({
				name: param.name,
				kind: 'expr',
				template: script,
				strategy: {
					execution: { mode: 'once-per-turn' },
					persistency: { kind: 'snapshot' },
					retry: 0,
				},
				output_ty: ty
			});
			sideEffects.push(param.name);
		} else if (param.resolution.kind === 'dynamic') {
			if (ty) {
				context[param.name] = { type: ty };
			} else {
				context[param.name] = { type: { kind: 'primitive', name: 'string' } };
			}
		} else if (param.resolution.kind === 'unresolved') {
			errors.push(`param '${param.name}': unresolved (set to static or dynamic)`);
		}
	}

	// Display nodes — convert bot display + regions into NodeConfigs.
	if (bot.display && bot.display.iterator?.trim() && bot.display.entries?.length > 0) {
		for (const entry of bot.display.entries) {
			const displayNodeName = `__display_${entry.name || 'main'}`;
			nodeConfigs.push({
				name: displayNodeName,
				kind: 'display',
				iterator: bot.display.iterator,
				template: entry.template,
				strategy: {
					execution: { mode: 'once-per-turn' },
					persistency: { kind: 'ephemeral' },
					retry: 0,
				},
			});
		}
	}

	// Region display nodes
	if (bot.regions) {
		for (const region of bot.regions) {
			if (region.kind === 'iterable' && region.iterator?.trim()) {
				for (const entry of region.entries ?? []) {
					const regionNodeName = `__region_${region.id}_${entry.name || 'main'}`;
					nodeConfigs.push({
						name: regionNodeName,
						kind: 'display',
						iterator: region.iterator,
						template: entry.template,
						strategy: {
							execution: { mode: 'once-per-turn' },
							persistency: { kind: 'ephemeral' },
							retry: 0,
						},
					});
				}
			} else if (region.kind === 'static' && region.template?.trim()) {
				const regionNodeName = `__region_${region.id}`;
				nodeConfigs.push({
					name: regionNodeName,
					kind: 'display_static',
					template: region.template,
					strategy: {
						execution: { mode: 'once-per-turn' },
						persistency: { kind: 'ephemeral' },
						retry: 0,
					},
				});
			}
		}
	}

	// Entrypoint: must be a node named "entrypoint".
	const entrypoint = allNodes.find(n => n.name === 'entrypoint')?.name;
	if (!entrypoint) {
		errors.push("no node named 'entrypoint': create one to serve as the main execution entry");
	}

	// Validate inputParam is a dynamic context param
	if (prompt.inputParam) {
		const inputParamDef = allActiveParams.find((p) => p.name === prompt.inputParam);
		if (!inputParamDef) {
			errors.push(`inputParam '${prompt.inputParam}' is not a known context param`);
		} else if (inputParamDef.resolution.kind !== 'dynamic') {
			errors.push(`inputParam '${prompt.inputParam}' must be dynamic (currently ${inputParamDef.resolution.kind})`);
		}
	}

	if (errors.length > 0) {
		return { ok: false, errors };
	}

	return {
		ok: true,
		config: {
			nodes: nodeConfigs,
			providers,
			entrypoint: entrypoint!,
			context,
			side_effects: sideEffects,
			asset_store_name: `asset_${bot.id}`,
		}
	};
}
