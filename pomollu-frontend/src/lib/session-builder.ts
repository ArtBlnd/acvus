import type { Bot, Prompt, Profile, Node, MessageDef, Block, RawBlock, ContextBlock, ContextBinding } from './types.js';
import { isRawBlock, isContextBlock, CONTEXT_TYPE } from './types.js';
import type { SessionConfig, NodeConfig, MessageConfig, ProviderConfig, StrategyConfig } from './engine.js';
import type { TypeDesc } from './type-parser.js';
import { isUnknownType } from './type-parser.js';
import { collectNodes, collectBlocks } from './block-tree.js';
import { promptStore, profileStore, providerStore } from './stores.svelte.js';
import { analyzeBot } from './param-resolver.js';

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

function convertNode(
	node: Node,
	blockLookup: Map<string, RawBlock>,
	allNodes: Node[],
	errors: string[]
): NodeConfig {
	if (node.kind === 'expr') {
		return {
			name: node.name,
			kind: 'expr',
			template: node.exprSource,
			initial_value: node.selfSpec.initialValue?.trim() || undefined,
			strategy: convertStrategy(node.strategy),
			retry: node.retry,
			assert_script: node.assert?.trim() || undefined,
			is_function: node.isFunction || undefined,
			fn_params: node.isFunction ? node.fnParams.map(p => ({ name: p.name, type: p.type })) : undefined
		};
	}

	const provider = node.kind === 'llm' ? providerStore.get(node.providerId) : undefined;

	if (node.kind === 'llm' && !provider) {
		errors.push(`node '${node.name}': provider not found`);
	}

	// Resolve tool bindings — each tool references another node by id
	const tools = node.tools.map((t) => {
		const targetNode = allNodes.find((n) => n.id === t.nodeId);
		if (!targetNode) {
			errors.push(`node '${node.name}': tool '${t.name}' references unknown node`);
		}
		return {
			name: t.name,
			description: t.description,
			node: targetNode?.name ?? '',
			params: Object.fromEntries(t.params.map((p) => [p.name, p.type]))
		};
	}).filter((t) => t.node !== '');

	return {
		name: node.name,
		kind: node.kind,
		initial_value: node.selfSpec.initialValue?.trim() || undefined,
		strategy: convertStrategy(node.strategy),
		retry: node.retry,
		assert_script: node.assert?.trim() || undefined,
		provider: provider?.name ?? '',
		api: provider?.api,
		model: node.model,
		temperature: node.temperature,
		top_p: node.topP,
		top_k: node.topK,
		grounding: node.grounding,
		max_tokens: { input: node.maxTokens.input, output: node.maxTokens.output },
		messages: node.messages.map((m) => convertMessage(m, blockLookup)),
		tools,
		is_function: node.isFunction || undefined,
		fn_params: node.isFunction ? node.fnParams.map(p => ({ name: p.name, type: p.type })) : undefined
	};
}

function convertStrategy(strategy: Node['strategy']): StrategyConfig {
	switch (strategy.mode) {
		case 'always':
			return { mode: 'always' };
		case 'once-per-turn':
			return { mode: 'once-per-turn' };
		case 'if-modified':
			return { mode: 'if-modified', key: strategy.key };
		case 'history':
			return { mode: 'history', history_bind: strategy.historyBind };
	}
}

/** Convert a Prompt contextBinding to an Expr NodeConfig. */
function bindingToExprNode(binding: ContextBinding): NodeConfig {
	return {
		name: binding.name,
		kind: 'expr',
		template: binding.script,
		strategy: { mode: 'once-per-turn' },
		retry: 0
	};
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
function buildContextNodes(allBlocks: Block[]): NodeConfig[] {
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
				strategy: { mode: 'once-per-turn' },
				retry: 0
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

	nodes.push({
		name: 'context',
		kind: 'expr',
		template: `{${fields.join(', ')},}`,
		strategy: { mode: 'once-per-turn' },
		retry: 0,
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

	// Convert user-defined nodes → NodeConfigs
	const nodeConfigs: NodeConfig[] = allNodes.map((n) => convertNode(n, blockLookup, allNodes, errors));

	// Prompt contextBindings → Expr nodes (also tracked as side effects)
	const sideEffects: string[] = [];
	for (const binding of prompt.contextBindings) {
		if (binding.name && binding.script.trim()) {
			if (seenNodeNames.has(binding.name)) {
				errors.push(`binding '${binding.name}' conflicts with a node name`);
			}
			seenNodeNames.add(binding.name);
			nodeConfigs.push(bindingToExprNode(binding));
			sideEffects.push(binding.name);
		}
	}

	// ContextBlocks → @context object grouped by type
	nodeConfigs.push(...buildContextNodes(allBlocks));

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

	// analyzeBot typechecks all 3 levels and returns ALL params merged
	// (prompt + profile + bot). This is the single source of truth.
	const analysisResult = analyzeBot(bot, prompt, profile, (id) => providerStore.get(id)?.api ?? '');
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
				strategy: { mode: 'always' },
				retry: 0,
				output_ty: ty
			});
			sideEffects.push(param.name);
		} else if (param.resolution.kind === 'dynamic') {
			if (ty) {
				context[param.name] = { type: ty };
			} else {
				context[param.name] = { type: { kind: 'primitive', name: 'String' } };
			}
		} else if (param.resolution.kind === 'unresolved') {
			errors.push(`param '${param.name}': unresolved (set to static or dynamic)`);
		}
	}

	// Entrypoint: the single history-strategy node. If none, last user-defined node.
	const historyNodes = allNodes.filter((n) => n.strategy.mode === 'history');
	if (historyNodes.length > 1) {
		errors.push(`multiple history nodes: ${historyNodes.map((n) => n.name).join(', ')} (only one allowed)`);
	}
	const entrypoint = historyNodes[0]?.name
		?? allNodes[allNodes.length - 1]?.name
		?? nodeConfigs[nodeConfigs.length - 1]?.name;
	if (!entrypoint) {
		errors.push('no entrypoint: add at least one node');
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
			entrypoint,
			context,
			side_effects: sideEffects
		}
	};
}
