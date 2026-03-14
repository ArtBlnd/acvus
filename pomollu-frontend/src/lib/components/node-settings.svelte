<script lang="ts">
	import type { Node, MessageDef, Execution, Persistency, FnParam, NodeToolBinding } from '$lib/types.js';
	import { blockLabel } from '$lib/types.js';
	import type { BlockOwner } from '$lib/stores.svelte.js';
	import type { DiscoveredFnParam } from '$lib/param-resolver.js';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { Button } from '$lib/components/ui/button';
	import * as Select from '$lib/components/ui/select';
	import { getOwnerChildren, updateOwnerNodeItem, uiState, providerStore, createRawBlock, generateName, collectScopeFunctionNodes } from '$lib/stores.svelte.js';
	import { findNodeItem, collectBlocks, collectAllNames } from '$lib/block-tree.js';
	import { Plus, Trash2 } from 'lucide-svelte';
	import { Switch } from '$lib/components/ui/switch';
	import AcvusEngineField from './acvus-engine-field.svelte';
	import BasePage from './base-page.svelte';
	import { collectOwnerDeps } from '$lib/dependencies.js';
	import { formatErrors, type NodeErrors } from '$lib/engine.js';
	import { isUnknownType, typeDescToString, parseTypeDesc } from '$lib/type-parser.js';
	import { typeDescToFnParamString } from '$lib/param-resolver.js';

	let {
		nodeId,
		owner,
		contextTypes = {},
		nodeLocals = {},
		nodeErrors = {},
		nodeFnParams = {},
	}: {
		nodeId: string;
		owner: BlockOwner;
		contextTypes?: Record<string, import('$lib/type-parser.js').TypeDesc>;
		nodeLocals?: Record<string, { raw: import('$lib/type-parser.js').TypeDesc; self: import('$lib/type-parser.js').TypeDesc }>;
		nodeErrors?: Record<string, NodeErrors>;
		nodeFnParams?: Record<string, DiscoveredFnParam[]>;
	} = $props();

	let node = $derived.by(() => {
		const children = getOwnerChildren(owner);
		return children ? findNodeItem(children, nodeId) : undefined;
	});

	// Merge external context types with this node's local types (@raw, @self) and fn_params
	let locals = $derived(node ? nodeLocals[node.name] : undefined);
	let mergedContextTypes = $derived.by((): Record<string, import('$lib/type-parser.js').TypeDesc> => {
		const base: Record<string, import('$lib/type-parser.js').TypeDesc> = locals
			? { ...contextTypes, raw: locals.raw, self: locals.self }
			: { ...contextTypes };
		if (!node?.isFunction) return base;
		// Inject fn_params so inline typecheck sees @a, @b, etc.
		const discovered = nodeFnParams[node.name] ?? [];
		for (const dp of discovered) {
			const storedType = node.fnParams.find(fp => fp.name === dp.name)?.type;
			if (storedType) {
				base[dp.name] = parseTypeDesc(storedType);
			} else if (!isUnknownType(dp.inferredType)) {
				base[dp.name] = dp.inferredType;
			}
		}
		return base;
	});

	// Per-field errors from hard typecheck (Phase 2)
	const EMPTY_NODE_ERRORS: NodeErrors = { initialValue: [], bind: [], ifModifiedKey: [], assert: [], messages: {}, exprSource: [] };
	let fieldErrors = $derived(node ? (nodeErrors[node.name] ?? EMPTY_NODE_ERRORS) : EMPTY_NODE_ERRORS);
	let providers = $derived(providerStore.providers);
	let hasOrphanProvider = $derived(
		node?.kind === 'llm' && node.providerId !== '' && !providerStore.get(node.providerId)
	);

	function updateNode(updater: (n: Node) => Node) {
		updateOwnerNodeItem(owner, nodeId, updater);
	}

	function handleRemove() {
		uiState.removeOwnerTreeNode(owner, nodeId);
	}

	// --- Strategy (Execution + Persistency + InitialValue + Retry + Assert) ---

	const execution = $derived(node?.strategy.execution ?? { mode: 'once-per-turn' as const });

	function setExecutionMode(mode: string) {
		const executions: Record<string, Execution> = {
			'always': { mode: 'always' },
			'once-per-turn': { mode: 'once-per-turn' },
			'if-modified': { mode: 'if-modified', key: '' },
		};
		updateNode((n) => ({ ...n, strategy: { ...n.strategy, execution: executions[mode] } }));
	}

	const persistency = $derived(node?.strategy.persistency ?? { kind: 'ephemeral' as const });

	function setPersistencyKind(kind: string) {
		const persistencies: Record<string, Persistency> = {
			'ephemeral': { kind: 'ephemeral' },
			'snapshot': { kind: 'snapshot' },
			'deque': { kind: 'deque', bind: '' },
			'diff': { kind: 'diff', bind: '' },
		};
		updateNode((n) => ({ ...n, strategy: { ...n.strategy, persistency: persistencies[kind] } }));
	}

	const persistencyLabels: Record<string, string> = {
		'ephemeral': 'Ephemeral',
		'snapshot': 'Snapshot',
		'deque': 'Deque',
		'diff': 'Diff',
	};

	const persistencyDescriptions: Record<string, string> = {
		'ephemeral': 'Output is not persisted to storage.',
		'snapshot': 'Overwrite stored value entirely each time.',
		'deque': 'Tracked deque. Requires a bind script.',
		'diff': 'Object field-level patch. Requires a bind script.',
	};

	// --- Messages ---

	function addBlockMessage() {
		updateNode((n) => ({
			...n,
			messages: [...(n.messages ?? []), { kind: 'block', role: 'system', source: { type: 'inline', template: '' } }]
		}));
	}

	function addIteratorMessage() {
		updateNode((n) => ({
			...n,
			messages: [...(n.messages ?? []), { kind: 'iterator', iterator: '' }]
		}));
	}

	function updateMessage(index: number, patch: Record<string, unknown>) {
		updateNode((n) => {
			const msgs = [...(n.messages ?? [])];
			msgs[index] = { ...msgs[index], ...patch } as MessageDef;
			return { ...n, messages: msgs };
		});
	}

	function setMessageSource(index: number, source: import('$lib/types.js').MessageSource) {
		updateNode((n) => {
			const msgs = [...(n.messages ?? [])];
			const m = msgs[index];
			if (m.kind !== 'block') return n;
			msgs[index] = { ...m, source };
			return { ...n, messages: msgs };
		});
	}

	function removeMessage(index: number) {
		updateNode((n) => ({
			...n,
			messages: (n.messages ?? []).filter((_, i) => i !== index)
		}));
	}

	function moveMessage(index: number, direction: -1 | 1) {
		updateNode((n) => {
			const msgs = [...(n.messages ?? [])];
			const target = index + direction;
			if (target < 0 || target >= msgs.length) return n;
			[msgs[index], msgs[target]] = [msgs[target], msgs[index]];
			return { ...n, messages: msgs };
		});
	}

	function addBlockToNode(): string {
		const existing = node ? collectAllNames(node.children) : [];
		const name = generateName('Block', existing);
		const newBlock = createRawBlock(name);
		updateNode((n) => ({
			...n,
			children: [...n.children, { kind: 'block' as const, block: newBlock }]
		}));
		return newBlock.id;
	}

	const messages = $derived(node?.messages ?? []);

	let collapsed: Record<string, boolean> = $state({});

	// All blocks in the owner's tree (including node children) for block reference picker
	const availableBlocks = $derived.by(() => {
		const ownerChildren = getOwnerChildren(owner);
		return ownerChildren ? collectBlocks(ownerChildren) : [];
	});

	function toggleSection(name: string) {
		collapsed[name] = !collapsed[name];
	}

	const roleColors: Record<string, string> = {
		system: 'msg-system',
		user: 'msg-user',
		assistant: 'msg-assistant',
	};

	function msgRole(msg: MessageDef): string | undefined {
		return msg.kind === 'block' ? msg.role : msg.role;
	}

	const executionLabels: Record<string, string> = {
		'always': 'Always',
		'once-per-turn': 'Once Per Turn',
		'if-modified': 'If Modified',
	};

	const executionDescriptions: Record<string, string> = {
		'always': 'Execute every invocation. @self is volatile (per-turn).',
		'once-per-turn': 'Execute once per turn. @self is persistent.',
		'if-modified': 'Execute only when key changes. @self is persistent.',
	};

	let deps = $derived(collectOwnerDeps(owner));

	let discoveredFnParams = $derived(node?.isFunction ? (nodeFnParams[node.name] ?? []) : []);

	// --- Tool bindings (LLM only) ---

	let availableFnNodes = $derived(node ? collectScopeFunctionNodes(owner, node.id) : []);

	function addToolBinding() {
		updateNode((n) => ({
			...n,
			tools: [...n.tools, { nodeId: '', description: '' }]
		}));
	}

	function updateToolBinding(index: number, patch: Partial<NodeToolBinding>) {
		updateNode((n) => ({
			...n,
			tools: n.tools.map((t, i) => i === index ? { ...t, ...patch } : t)
		}));
	}

	function removeToolBinding(index: number) {
		updateNode((n) => ({
			...n,
			tools: n.tools.filter((_, i) => i !== index)
		}));
	}

	function getToolTargetNode(nodeId: string): Node | undefined {
		return availableFnNodes.find((n) => n.id === nodeId);
	}

	// Sync fnParams with discovered params: add new, remove stale, preserve user overrides.
	$effect(() => {
		if (!node) return;
		const liveNames = new Set(discoveredFnParams.map(dp => dp.name));
		const currentNames = new Set((node.fnParams ?? []).map(fp => fp.name));
		const stale = (node.fnParams ?? []).filter(fp => !liveNames.has(fp.name));
		const added = discoveredFnParams.filter(dp => !currentNames.has(dp.name));
		if (stale.length > 0 || added.length > 0) {
			updateNode((n) => {
				const kept = (n.fnParams ?? []).filter(fp => liveNames.has(fp.name));
				const newEntries = added.map(dp => ({
					name: dp.name,
					type: typeDescToFnParamString(dp.inferredType),
					description: undefined as string | undefined,
				}));
				return { ...n, fnParams: [...kept, ...newEntries] };
			});
		}
	});
</script>

<BasePage {deps} onConfigChange={() => {}}>
	<div class="flex items-center justify-between shrink-0 border-b px-3 py-2">
		<span class="text-sm font-medium">Node Settings</span>
		<Button variant="ghost" size="icon-sm" class="text-muted-foreground hover:text-destructive" onclick={handleRemove} title="Delete node">
			&times;
		</Button>
	</div>

	{#if node}
		<div class="flex-1 overflow-y-auto min-h-0 {node.kind === 'expr' ? 'flex flex-col' : ''}">
			<div class="sections {node.kind === 'expr' ? 'flex-1 min-h-0' : ''}">
				<!-- ========== BASIC ========== -->
				<section class="section">
					<button class="section-header" onclick={() => toggleSection('basic')}>
						<svg class="section-chevron" class:collapsed={collapsed['basic']} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9l6 6 6-6"/></svg>
						<span>Basic</span>
					</button>
					{#if !collapsed['basic']}<div class="section-body">
						<div class="field">
							<Label>Name</Label>
							<Input
								value={node.name}
								oninput={(e) => updateNode((n) => ({ ...n, name: e.currentTarget.value }))}
							/>
						</div>
						<div class="field">
							<Label>Kind</Label>
							<Select.Root type="single" value={node.kind} onValueChange={(v) => updateNode((n) => ({ ...n, kind: v as import('$lib/types.js').NodeKind }))}>
								<Select.Trigger class="w-full">{node.kind === 'llm' ? 'LLM' : node.kind === 'expr' ? 'Expression' : 'Plain'}</Select.Trigger>
								<Select.Content>
									<Select.Item value="llm">LLM</Select.Item>
									<Select.Item value="plain">Plain</Select.Item>
									<Select.Item value="expr">Expression</Select.Item>
								</Select.Content>
							</Select.Root>
						</div>
						<div class="flex items-center gap-2">
							<Switch
								checked={node.isFunction}
								onCheckedChange={(v) => updateNode((n) => ({ ...n, isFunction: v }))}
							/>
							<Label class="text-sm">Function</Label>
						</div>
						{#if node.isFunction}
							<div class="field">
								<Label>Parameters</Label>
								{#if discoveredFnParams.length === 0}
									<p class="text-xs text-muted-foreground italic">No parameters discovered. Use @name in the node body to declare arguments.</p>
								{:else}
									<div class="fn-params-list">
										{#each discoveredFnParams as dp (dp.name)}
											{@const stored = node.fnParams.find(fp => fp.name === dp.name)}
											{@const storedType = stored?.type}
											{@const hasInferred = !isUnknownType(dp.inferredType)}
											{@const displayType = storedType || (hasInferred ? typeDescToString(dp.inferredType) : '')}
											<div class="fn-param-row">
												<code class="fn-param-name">@{dp.name}</code>
												{#if hasInferred && !storedType}
													<span class="fn-param-inferred">{typeDescToString(dp.inferredType)}</span>
												{:else}
													<Input
														class="fn-param-input"
														placeholder="type..."
														value={displayType}
														oninput={(e) => {
															const val = e.currentTarget.value.trim();
															const params = discoveredFnParams.map(p => {
																if (p.name === dp.name) return { name: p.name, type: val, description: stored?.description };
																const existing = node.fnParams.find(fp => fp.name === p.name);
																return { name: p.name, type: existing?.type || '', description: existing?.description };
															});
															updateNode((n) => ({ ...n, fnParams: params }));
														}}
													/>
												{/if}
											</div>
											<div class="fn-param-desc-row">
												<Input
													class="fn-param-desc-input"
													placeholder="description..."
													value={stored?.description ?? ''}
													oninput={(e) => {
														const val = e.currentTarget.value;
														const params = discoveredFnParams.map(p => {
															const existing = node.fnParams.find(fp => fp.name === p.name);
															if (p.name === dp.name) return { name: p.name, type: existing?.type || displayType, description: val || undefined };
															return { name: p.name, type: existing?.type || '', description: existing?.description };
														});
														updateNode((n) => ({ ...n, fnParams: params }));
													}}
												/>
											</div>
										{/each}
									</div>
								{/if}
								<p class="hint">Callable as @{node.name}(args...) from other nodes.</p>
							</div>
						{/if}
						{#if node.kind === 'llm'}
							<div class="field">
								<Label>Provider</Label>
								<Select.Root type="single" value={node.providerId} onValueChange={(v) => updateNode((n) => ({ ...n, providerId: v }))}>
									<Select.Trigger class="w-full {hasOrphanProvider ? 'border-destructive' : ''}">
										{providerStore.get(node.providerId)?.name ?? 'Select provider...'}
									</Select.Trigger>
									<Select.Content>
										{#each providers as p (p.id)}
											<Select.Item value={p.id}>{p.name}</Select.Item>
										{/each}
									</Select.Content>
								</Select.Root>
								{#if hasOrphanProvider}
									<p class="text-xs text-destructive">Provider has been deleted.</p>
								{/if}
							</div>
							<div class="field">
								<Label>Model</Label>
								<Input
									value={node.model}
									oninput={(e) => updateNode((n) => ({ ...n, model: e.currentTarget.value }))}
									placeholder="gemini-2.5-flash, gpt-4o, ..."
								/>
							</div>
							<div class="grid grid-cols-3 gap-2">
								<div class="field">
									<Label>Temperature</Label>
									<Input
										type="number"
										step="0.1"
										min="0"
										max="2"
										value={String(node.temperature)}
										oninput={(e) => updateNode((n) => ({ ...n, temperature: Math.round(Number(e.currentTarget.value) * 100) / 100 || 0 }))}
									/>
								</div>
								<div class="field">
									<Label>Top P</Label>
									<Input
										type="number"
										step="0.05"
										min="0"
										max="1"
										value={node.topP != null ? String(node.topP) : ''}
										oninput={(e) => {
											const v = e.currentTarget.value.trim();
											const num = Number(v);
											updateNode((n) => ({ ...n, topP: v === '' ? null : num ? Math.round(num * 100) / 100 : null }));
										}}
										placeholder="—"
									/>
								</div>
								<div class="field">
									<Label>Top K</Label>
									<Input
										type="number"
										step="1"
										min="0"
										value={node.topK != null ? String(node.topK) : ''}
										oninput={(e) => {
											const v = e.currentTarget.value.trim();
											updateNode((n) => ({ ...n, topK: v === '' ? null : Math.round(Number(v)) || null }));
										}}
										placeholder="—"
									/>
								</div>
							</div>
							<div class="flex items-center gap-2">
								<Switch
									checked={node.grounding}
									onCheckedChange={(v) => updateNode((n) => ({ ...n, grounding: v }))}
								/>
								<Label class="text-sm">Grounding</Label>
							</div>
							<div class="field">
								<Label>Thinking</Label>
								<div class="flex gap-2">
									<Select.Root
										type="single"
										value={node.thinking?.kind ?? 'none'}
										onValueChange={(v) => {
											if (v === 'none') {
												updateNode((n) => ({ ...n, thinking: null }));
											} else if (v === 'custom') {
												updateNode((n) => ({ ...n, thinking: { kind: 'custom', value: (n.thinking as any)?.value ?? 4096 } }));
											} else {
												updateNode((n) => ({ ...n, thinking: { kind: v as 'off' | 'low' | 'medium' | 'high' } }));
											}
										}}
									>
										<Select.Trigger class="flex-1">{{ none: 'None', off: 'Off', low: 'Low', medium: 'Medium', high: 'High', custom: 'Custom' }[node.thinking?.kind ?? 'none']}</Select.Trigger>
										<Select.Content>
											<Select.Item value="none">None</Select.Item>
											<Select.Item value="off">Off</Select.Item>
											<Select.Item value="low">Low</Select.Item>
											<Select.Item value="medium">Medium</Select.Item>
											<Select.Item value="high">High</Select.Item>
											<Select.Item value="custom">Custom</Select.Item>
										</Select.Content>
									</Select.Root>
									{#if node.thinking?.kind === 'custom'}
										<Input
											type="number"
											class="w-28"
											min="1"
											value={String(node.thinking.value)}
											oninput={(e) => updateNode((n) => ({ ...n, thinking: { kind: 'custom', value: Number(e.currentTarget.value) || 1 } }))}
											placeholder="Budget"
										/>
									{/if}
								</div>
							</div>
							<div class="grid grid-cols-2 gap-2">
								<div class="field">
									<Label>Max Input Tokens</Label>
									<Input
										type="number"
										value={String(node.maxTokens.input)}
										oninput={(e) => updateNode((n) => ({ ...n, maxTokens: { ...n.maxTokens, input: Number(e.currentTarget.value) || 0 } }))}
									/>
								</div>
								<div class="field">
									<Label>Max Output Tokens</Label>
									<Input
										type="number"
										value={String(node.maxTokens.output)}
										oninput={(e) => updateNode((n) => ({ ...n, maxTokens: { ...n.maxTokens, output: Number(e.currentTarget.value) || 0 } }))}
									/>
								</div>
							</div>
						{/if}
					</div>{/if}
				</section>

				<!-- ========== STRATEGY ========== -->
				<section class="section">
					<button class="section-header" onclick={() => toggleSection('strategy')}>
						<svg class="section-chevron" class:collapsed={collapsed['strategy']} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9l6 6 6-6"/></svg>
						<span>Strategy</span>
					</button>
					{#if !collapsed['strategy']}<div class="section-body">
						<!-- Execution subsection -->
						<div class="field">
							<Label>Execution</Label>
							<Select.Root type="single" value={execution.mode} onValueChange={(v) => setExecutionMode(v)}>
								<Select.Trigger class="w-full">{executionLabels[execution.mode]}</Select.Trigger>
								<Select.Content>
									<Select.Item value="always">Always</Select.Item>
									<Select.Item value="once-per-turn">Once Per Turn</Select.Item>
									<Select.Item value="if-modified">If Modified</Select.Item>
								</Select.Content>
							</Select.Root>
							<p class="hint">{executionDescriptions[execution.mode]}</p>
						</div>
						{#if execution.mode === 'if-modified'}
							<div class="field">
								<Label>Key</Label>
								<AcvusEngineField
									mode="script"
									placeholder="e.g. @input | to_string"
									value={execution.key}
									oninput={(v) => updateNode((n) => ({ ...n, strategy: { ...n.strategy, execution: { mode: 'if-modified', key: v } } }))}
									contextTypes={mergedContextTypes}
									fieldError={formatErrors(fieldErrors.ifModifiedKey)}
									discoverContext
								/>
								<p class="hint">Script expression. Re-executes when this value changes.</p>
							</div>
						{/if}

						<!-- Persistency subsection -->
						<div class="field">
							<Label>Persistency</Label>
							<Select.Root type="single" value={persistency.kind} onValueChange={(v) => setPersistencyKind(v)}>
								<Select.Trigger class="w-full">{persistencyLabels[persistency.kind]}</Select.Trigger>
								<Select.Content>
									<Select.Item value="ephemeral">Ephemeral</Select.Item>
									<Select.Item value="snapshot">Snapshot</Select.Item>
									<Select.Item value="deque">Deque</Select.Item>
									<Select.Item value="diff">Diff</Select.Item>
								</Select.Content>
							</Select.Root>
							<p class="hint">{persistencyDescriptions[persistency.kind]}</p>
						</div>
						{#if persistency.kind === 'deque' || persistency.kind === 'diff'}
							<div class="field">
								<Label>Bind</Label>
								<AcvusEngineField
									mode="script"
									placeholder="e.g. @raw"
									value={persistency.bind}
									oninput={(v) => updateNode((n) => ({ ...n, strategy: { ...n.strategy, persistency: { ...n.strategy.persistency, bind: v } as Persistency } }))}
									contextTypes={mergedContextTypes}
									fieldError={formatErrors(fieldErrors.bind)}
									discoverContext
								/>
								<p class="hint">Script with access to @raw (node output) and @self (current stored value).</p>
							</div>
						{/if}

						<!-- Initial Value -->
						<div class="field">
							<Label>Initial Value</Label>
							<AcvusEngineField
								mode="script"
								placeholder=''
								value={node.strategy.initialValue ?? ''}
								oninput={(v) => updateNode((n) => ({ ...n, strategy: { ...n.strategy, initialValue: v } }))}
								contextTypes={mergedContextTypes}
								expectedTailType={locals?.self}
								fieldError={formatErrors(fieldErrors.initialValue)}
								discoverContext
							/>
							<p class="hint">Initial @self value. When set, @self is available in the node body (previous stored value or this initial value on first run). Leave empty to disable @self.</p>
						</div>

						<!-- Retry / Assert -->
						<div class="grid grid-cols-2 gap-2">
							<div class="field">
								<Label>Retry</Label>
								<Input
									type="number"
									min="0"
									value={String(node.strategy.retry ?? 0)}
									oninput={(e) => updateNode((n) => ({ ...n, strategy: { ...n.strategy, retry: Number(e.currentTarget.value) || 0 } }))}
								/>
							</div>
							<div class="field">
								<Label>Assert</Label>
								<AcvusEngineField
									mode="script"
									placeholder="e.g. @self | length > 0"
									value={node.strategy.assert ?? ''}
									oninput={(v) => updateNode((n) => ({ ...n, strategy: { ...n.strategy, assert: v } }))}
									contextTypes={mergedContextTypes}
									expectedTailType={{ kind: 'primitive', name: 'bool' }}
									fieldError={formatErrors(fieldErrors.assert)}
									discoverContext
								/>
							</div>
						</div>
					</div>{/if}
				</section>

				<!-- ========== MESSAGES (LLM only) ========== -->
				{#if node.kind === 'llm'}
					<section class="section">
						<div class="section-header">
							<button class="section-toggle" onclick={() => toggleSection('messages')}>
								<svg class="section-chevron" class:collapsed={collapsed['messages']} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9l6 6 6-6"/></svg>
								<span>Messages</span>
							</button>
							<div class="flex gap-1">
								<Button variant="outline" size="sm" class="h-6 text-xs" onclick={addBlockMessage}>
									<Plus class="mr-1 h-3 w-3" />
									Block
								</Button>
								<Button variant="outline" size="sm" class="h-6 text-xs" onclick={addIteratorMessage}>
									<Plus class="mr-1 h-3 w-3" />
									Iterator
								</Button>
							</div>
						</div>
						{#if !collapsed['messages']}<div class="section-body">
							{#if messages.length === 0}
								<div class="rounded-md border border-dashed p-4 text-center text-xs text-muted-foreground">
									No messages. Add a block or iterator message.
								</div>
							{/if}

							<div class="space-y-2">
								{#each messages as msg, i (i)}
									{@const role = msgRole(msg)}
									<div class="msg-card {role ? roleColors[role] ?? '' : ''}">
										<!-- Header -->
										<div class="flex items-center gap-1.5 px-2 py-1.5">
											<div class="flex flex-col gap-px">
												<button
													class="msg-move"
													disabled={i === 0}
													onclick={() => moveMessage(i, -1)}
													title="Move up"
												>
													<svg class="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 15l-6-6-6 6"/></svg>
												</button>
												<button
													class="msg-move"
													disabled={i === messages.length - 1}
													onclick={() => moveMessage(i, 1)}
													title="Move down"
												>
													<svg class="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9l6 6 6-6"/></svg>
												</button>
											</div>
											<span class="msg-kind-badge {msg.kind === 'iterator' ? 'msg-kind-iter' : ''}">
												{msg.kind === 'block' ? 'block' : 'iter'}
											</span>
											{#if msg.kind === 'block'}
												<Select.Root type="single" value={msg.role} onValueChange={(v) => updateMessage(i, { role: v })}>
													<Select.Trigger size="sm" class="msg-role-trigger {roleColors[msg.role] ?? ''}">{msg.role}</Select.Trigger>
													<Select.Content>
														<Select.Item value="system">system</Select.Item>
														<Select.Item value="user">user</Select.Item>
														<Select.Item value="assistant">assistant</Select.Item>
													</Select.Content>
												</Select.Root>
											{:else}
												{#if msg.role}
													<Select.Root type="single" value={msg.role} onValueChange={(v) => updateMessage(i, { role: v || undefined })}>
														<Select.Trigger size="sm" class="msg-role-trigger {roleColors[msg.role] ?? ''}">{msg.role}</Select.Trigger>
														<Select.Content>
															<Select.Item value="auto" label="auto">auto</Select.Item>
															<Select.Item value="system">system</Select.Item>
															<Select.Item value="user">user</Select.Item>
															<Select.Item value="assistant">assistant</Select.Item>
														</Select.Content>
													</Select.Root>
												{:else}
													<button
														class="pill-btn"
														onclick={() => updateMessage(i, { role: 'user' })}
													>+ role</button>
												{/if}
											{/if}
											<div class="flex-1"></div>
											<button
												class="rounded p-0.5 text-muted-foreground hover:text-destructive"
												onclick={() => removeMessage(i)}
												title="Remove"
											>
												<Trash2 class="h-3.5 w-3.5" />
											</button>
										</div>

										<!-- Body -->
										<div class="px-2 pb-2">
											{#if msg.kind === 'block'}
												{@const src = msg.source ?? { type: 'inline', template: '' }}
												<div class="space-y-1.5">
													<!-- Source toggle -->
													<div class="flex items-center gap-1.5">
														<button
															class="source-toggle {src.type === 'inline' ? 'active' : ''}"
															onclick={() => setMessageSource(i, { type: 'inline', template: '' })}
														>inline</button>
														<button
															class="source-toggle {src.type === 'block' ? 'active' : ''}"
															onclick={() => setMessageSource(i, { type: 'block', blockId: '' })}
														>block</button>
													</div>
													{#if src.type === 'inline'}
														<AcvusEngineField
															mode="template"
															placeholder="Template content..."
															value={src.template}
															oninput={(v) => setMessageSource(i, { type: 'inline', template: v })}
															contextTypes={mergedContextTypes}
															fieldError={formatErrors(fieldErrors.messages?.[String(i)])}
															discoverContext
														/>
													{:else}
														<div class="flex gap-1 items-center">
															<Select.Root type="single" value={src.blockId} onValueChange={(v) => setMessageSource(i, { type: 'block', blockId: v })}>
																<Select.Trigger class="w-full" size="sm">
																	{#if src.blockId}
																		{@const blk = availableBlocks.find(b => b.id === src.blockId)}
																		{blk ? blockLabel(blk) : 'Unknown'}
																	{:else}
																		Select block...
																	{/if}
																</Select.Trigger>
																<Select.Content>
																	{#each availableBlocks as b (b.id)}
																		<Select.Item value={b.id}>{blockLabel(b)}</Select.Item>
																	{/each}
																	{#if availableBlocks.length === 0}
																		<div class="px-2 py-1.5 text-xs text-muted-foreground">No blocks available</div>
																	{/if}
																</Select.Content>
															</Select.Root>
															<Button variant="outline" size="sm" class="h-7 shrink-0 text-xs" onclick={() => {
																const id = addBlockToNode();
																setMessageSource(i, { type: 'block', blockId: id });
															}}>New</Button>
														</div>
													{/if}
												</div>
											{:else}
												<div class="space-y-1.5">
													<AcvusEngineField
														mode="script"
														placeholder="Iterator expression"
														value={msg.iterator}
														oninput={(v) => updateMessage(i, { iterator: v })}
														contextTypes={mergedContextTypes}
														fieldError={formatErrors(fieldErrors.messages?.[String(i)])}
														discoverContext
													/>
													<!-- Slice -->
													{#if msg.slice}
														<div class="flex items-center gap-1.5">
															<span class="text-[10px] text-muted-foreground">slice</span>
															<input
																type="number"
																class="msg-slice-input"
																value={msg.slice[0]}
																oninput={(e) => {
																	const v = parseInt(e.currentTarget.value);
																	if (isNaN(v)) return;
																	const s = msg.slice!;
																	updateMessage(i, { slice: s.length > 1 ? [v, s[1]] : [v] });
																}}
																placeholder="start"
															/>
															{#if msg.slice.length > 1}
																<span class="text-[10px] text-muted-foreground">:</span>
																<input
																	type="number"
																	class="msg-slice-input"
																	value={msg.slice[1]}
																	oninput={(e) => {
																		const v = parseInt(e.currentTarget.value);
																		if (isNaN(v)) return;
																		updateMessage(i, { slice: [msg.slice![0], v] });
																	}}
																	placeholder="end"
																/>
															{:else}
																<button
																	class="pill-btn"
																	onclick={() => updateMessage(i, { slice: [msg.slice![0], -1] })}
																>+ end</button>
															{/if}
															<button
																class="rounded p-0.5 text-muted-foreground hover:text-destructive"
																onclick={() => updateMessage(i, { slice: undefined })}
																title="Remove slice"
															>
																<Trash2 class="h-3 w-3" />
															</button>
														</div>
													{:else}
														<button
															class="pill-btn"
															onclick={() => updateMessage(i, { slice: [0] })}
														>+ slice</button>
													{/if}
													<!-- Template -->
													{#if msg.template !== undefined}
														<AcvusEngineField
															mode="template"
															placeholder="Item template (optional)..."
															value={msg.template ?? ''}
															oninput={(v) => updateMessage(i, { template: v })}
															contextTypes={mergedContextTypes}
															discoverContext
														/>
													{:else}
														<button
															class="pill-btn"
															onclick={() => updateMessage(i, { template: '' })}
														>+ template</button>
													{/if}
												</div>
											{/if}
										</div>
									</div>
								{/each}
							</div>
						</div>{/if}
					</section>
				{/if}

				<!-- ========== TOOLS (LLM only) ========== -->
				{#if node.kind === 'llm'}
					<section class="section">
						<div class="section-header">
							<button class="section-toggle" onclick={() => toggleSection('tools')}>
								<svg class="section-chevron" class:collapsed={collapsed['tools']} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9l6 6 6-6"/></svg>
								<span>Tools</span>
							</button>
							<Button variant="outline" size="sm" class="h-6 text-xs" onclick={addToolBinding} disabled={availableFnNodes.length === 0}>
								<Plus class="mr-1 h-3 w-3" />
								Bind
							</Button>
						</div>
						{#if !collapsed['tools']}<div class="section-body">
							{#if node.tools.length === 0}
								<div class="rounded-md border border-dashed p-4 text-center text-xs text-muted-foreground">
									{#if availableFnNodes.length === 0}
										No function nodes available. Mark a node as Function to bind it as a tool.
									{:else}
										No tool bindings. Click Bind to add one.
									{/if}
								</div>
							{/if}

							<div class="space-y-2">
								{#each node.tools as tool, i (i)}
									{@const target = getToolTargetNode(tool.nodeId)}
									{@const isOrphan = tool.nodeId !== '' && !target}
									<div class="tool-card" class:tool-orphan={isOrphan}>
										<div class="flex items-center gap-2 px-2 py-1.5">
											<Select.Root
												type="single"
												value={tool.nodeId}
												onValueChange={(v) => updateToolBinding(i, { nodeId: v })}
											>
												<Select.Trigger class="flex-1 text-sm {isOrphan ? 'border-destructive' : ''}">
													{#if target}
														{target.name}
													{:else if isOrphan}
														<span class="text-destructive">Deleted node</span>
													{:else}
														<span class="text-muted-foreground">Select function...</span>
													{/if}
												</Select.Trigger>
												<Select.Content>
													{#each availableFnNodes as fn (fn.id)}
														<Select.Item value={fn.id}>{fn.name}</Select.Item>
													{/each}
													{#if availableFnNodes.length === 0}
														<div class="px-2 py-1.5 text-xs text-muted-foreground">No function nodes</div>
													{/if}
												</Select.Content>
											</Select.Root>
											<button
												class="rounded p-0.5 text-muted-foreground hover:text-destructive"
												onclick={() => removeToolBinding(i)}
												title="Remove"
											>
												<Trash2 class="h-3.5 w-3.5" />
											</button>
										</div>
										<div class="px-2 pb-2 space-y-1.5">
											<Input
												class="text-xs"
												placeholder="Description (optional)..."
												value={tool.description}
												oninput={(e) => updateToolBinding(i, { description: e.currentTarget.value })}
											/>
											{#if target && target.fnParams.length > 0}
												<div class="tool-params">
													{#each target.fnParams as param (param.name)}
														<div class="tool-param-row">
															<code class="tool-param-name">@{param.name}</code>
															<span class="tool-param-type">{param.type}</span>
														</div>
													{/each}
												</div>
											{/if}
											{#if isOrphan}
												<p class="text-xs text-destructive">Target function node has been deleted.</p>
											{/if}
										</div>
									</div>
								{/each}
							</div>
						</div>{/if}
					</section>
				{/if}

				<!-- ========== EXPR SCRIPT (Expr only) ========== -->
				{#if node.kind === 'expr'}
					<section class="section" style="flex: 1; display: flex; flex-direction: column; min-height: 0;">
						<div class="section-body" style="flex: 1; display: flex; flex-direction: column; min-height: 0;">
							<Label>Script</Label>
							<AcvusEngineField
								mode="script"
								placeholder="Acvus script expression..."
								value={node.exprSource}
								oninput={(v) => updateNode((n) => ({ ...n, exprSource: v }))}
								contextTypes={mergedContextTypes}
								fieldError={formatErrors(fieldErrors?.exprSource)}
								discoverContext
								unlimited
							/>
						</div>
					</section>
				{/if}
			</div>
		</div>
	{:else}
		<div class="flex flex-1 items-center justify-center text-sm text-muted-foreground">
			No node selected.
		</div>
	{/if}
</BasePage>

<style>
	/* Sections */
	.sections {
		display: flex;
		flex-direction: column;
		padding-top: 0.5rem;
		padding-bottom: 4rem;
	}
	.section {
		border-bottom: 1px solid var(--color-border);
	}
	.section:last-child {
		border-bottom: none;
	}
	.section-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0.5rem 1rem;
		font-size: 0.6875rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--color-muted-foreground);
		background: var(--color-muted);
		cursor: pointer;
		user-select: none;
	}
	/* When section-header is a button (Basic, Execution) */
	button.section-header {
		width: 100%;
		border: none;
		text-align: left;
		justify-content: flex-start;
		gap: 0.375rem;
	}
	/* Toggle part inside Messages header */
	.section-toggle {
		display: flex;
		align-items: center;
		gap: 0.375rem;
		border: none;
		background: none;
		padding: 0;
		font: inherit;
		color: inherit;
		text-transform: inherit;
		letter-spacing: inherit;
		cursor: pointer;
	}
	.section-chevron {
		width: 0.75rem;
		height: 0.75rem;
		transition: transform 0.15s;
		flex-shrink: 0;
	}
	.section-chevron.collapsed {
		transform: rotate(-90deg);
	}
	.section-body {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
		padding: 0.75rem 1rem;
	}

	/* Field */
	.field {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	/* Hint text */
	.hint {
		font-size: 0.6875rem;
		color: var(--color-muted-foreground);
		line-height: 1.3;
	}

	/* Source toggle */
	.source-toggle {
		padding: 0.125rem 0.5rem;
		border-radius: 0.25rem;
		border: 1px solid var(--color-border);
		background: transparent;
		font-size: 0.625rem;
		font-weight: 500;
		color: var(--color-muted-foreground);
		cursor: pointer;
		transition: all 0.15s;
	}
	.source-toggle:hover {
		border-color: var(--color-foreground);
		color: var(--color-foreground);
	}
	.source-toggle.active {
		background: var(--color-foreground);
		color: var(--color-background);
		border-color: var(--color-foreground);
	}

	/* Pill button (+ role, + slice, + template) */
	.pill-btn {
		height: 1.5rem;
		border-radius: 9999px;
		border: 1px dashed var(--color-border);
		padding: 0 0.5rem;
		font-size: 0.625rem;
		color: var(--color-muted-foreground);
		background: transparent;
		cursor: pointer;
		transition: all 0.15s;
	}
	.pill-btn:hover {
		border-color: var(--color-foreground);
		color: var(--color-foreground);
	}

	/* Message card */
	.msg-card {
		border-radius: 0.5rem;
		border: 1px solid var(--color-border);
		border-left: 3px solid var(--color-border);
		background: var(--color-card);
		transition: border-color 0.15s;
	}
	.msg-card.msg-system { border-left-color: var(--color-msg-system, #f59e0b); }
	.msg-card.msg-user { border-left-color: var(--color-msg-user, #3b82f6); }
	.msg-card.msg-assistant { border-left-color: var(--color-msg-assistant, #10b981); }

	/* Role trigger */
	:global(.msg-role-trigger) {
		border-radius: 9999px !important;
		font-size: 0.6875rem !important;
		font-weight: 500 !important;
		height: 1.5rem !important;
		padding: 0 0.5rem !important;
		gap: 0.25rem !important;
	}
	:global(.msg-role-trigger.msg-system) {
		color: var(--color-msg-system, #f59e0b) !important;
		border-color: color-mix(in srgb, var(--color-msg-system, #f59e0b) 40%, transparent) !important;
		background: color-mix(in srgb, var(--color-msg-system, #f59e0b) 8%, transparent) !important;
	}
	:global(.msg-role-trigger.msg-user) {
		color: var(--color-msg-user, #3b82f6) !important;
		border-color: color-mix(in srgb, var(--color-msg-user, #3b82f6) 40%, transparent) !important;
		background: color-mix(in srgb, var(--color-msg-user, #3b82f6) 8%, transparent) !important;
	}
	:global(.msg-role-trigger.msg-assistant) {
		color: var(--color-msg-assistant, #10b981) !important;
		border-color: color-mix(in srgb, var(--color-msg-assistant, #10b981) 40%, transparent) !important;
		background: color-mix(in srgb, var(--color-msg-assistant, #10b981) 8%, transparent) !important;
	}

	/* Kind badge */
	.msg-kind-badge {
		border-radius: 0.25rem;
		padding: 0.125rem 0.375rem;
		font-size: 0.625rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.03em;
		color: var(--color-muted-foreground);
		background: var(--color-muted);
	}
	.msg-kind-iter {
		color: var(--color-msg-iter, #8b5cf6);
		background: color-mix(in srgb, var(--color-msg-iter, #8b5cf6) 12%, transparent);
	}

	/* Move buttons */
	.msg-move {
		color: var(--color-muted-foreground);
		transition: color 0.1s;
	}
	.msg-move:hover { color: var(--color-foreground); }
	.msg-move:disabled { opacity: 0.25; pointer-events: none; }

	/* Slice input */
	.msg-slice-input {
		width: 4.5rem;
		height: 1.5rem;
		border-radius: 0.25rem;
		border: 1px solid var(--color-border);
		background: transparent;
		padding: 0 0.375rem;
		font-family: var(--font-mono, ui-monospace, monospace);
		font-size: 0.6875rem;
		text-align: center;
	}
	.msg-slice-input:focus {
		outline: none;
		box-shadow: 0 0 0 1px var(--color-ring);
	}

	/* Fn params */
	.fn-params-list {
		display: flex;
		flex-direction: column;
		border-radius: 0.375rem;
		border: 1px solid var(--color-border);
		overflow: hidden;
	}
	.fn-param-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0.375rem 0.625rem;
		gap: 0.5rem;
	}
	.fn-param-row + .fn-param-row {
		border-top: 1px solid var(--color-border);
	}
	.fn-param-name {
		font-size: 0.75rem;
		font-weight: 600;
		color: var(--color-foreground);
		white-space: nowrap;
	}
	.fn-param-inferred {
		font-size: 0.6875rem;
		color: var(--color-muted-foreground);
		font-family: var(--font-mono, ui-monospace, monospace);
	}
	/* Tool cards */
	.tool-card {
		border-radius: 0.5rem;
		border: 1px solid var(--color-border);
		background: var(--color-card);
	}
	.tool-orphan {
		border-color: var(--color-destructive);
	}
	.tool-params {
		display: flex;
		flex-direction: column;
		border-radius: 0.25rem;
		border: 1px solid var(--color-border);
		overflow: hidden;
		background: var(--color-muted);
	}
	.tool-param-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0.25rem 0.5rem;
		gap: 0.5rem;
	}
	.tool-param-row + .tool-param-row {
		border-top: 1px solid var(--color-border);
	}
	.tool-param-name {
		font-size: 0.6875rem;
		font-weight: 600;
		color: var(--color-foreground);
		white-space: nowrap;
	}
	.tool-param-type {
		font-size: 0.625rem;
		color: var(--color-muted-foreground);
		font-family: var(--font-mono, ui-monospace, monospace);
	}

	.fn-param-desc-row {
		padding: 0 0.625rem 0.375rem;
	}
	:global(.fn-param-desc-input) {
		width: 100% !important;
		height: 1.5rem !important;
		font-size: 0.625rem !important;
		color: var(--color-muted-foreground) !important;
	}
	:global(.fn-param-input) {
		width: 6rem !important;
		height: 1.5rem !important;
		font-size: 0.6875rem !important;
		text-align: left !important;
		font-family: var(--font-mono, ui-monospace, monospace) !important;
	}

</style>
