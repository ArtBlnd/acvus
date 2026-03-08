import type { Block, ContextBlock, RawBlock, RawBlockMode, ScriptBlock, BlockNode, Node, Prompt, Profile, Provider, Bot, Session, NoneBlock } from './types.js';
import { createDefaultLayout, HISTORY_BINDING_NAME } from './types.js';
import { addNode, removeTreeNode, updateBlock as treeUpdateBlock, updateNodeItem as treeUpdateNodeItem, findTreeNode, findBlock, findNodeItem, collectAllIds, collectBlocks } from './block-tree.js';

export function createId(): string {
	return crypto.randomUUID();
}

// --- Block type registry ---

const defaultTypes = ['disabled', 'character', 'world_info', 'lorebook', 'memory'];

class BlockTypeRegistry {
	types = $state<string[]>([...defaultTypes]);

	add(type: string) {
		if (!this.types.includes(type)) {
			this.types = [...this.types, type];
		}
	}

	remove(type: string) {
		this.types = this.types.filter((t) => t !== type);
	}
}

export const blockTypeRegistry = new BlockTypeRegistry();

// --- Name generation ---

export function generateName(prefix: string, existingNames: Iterable<string>): string {
	const used = new Set(existingNames);
	if (!used.has(prefix)) return prefix;
	for (let i = 2; ; i++) {
		const name = `${prefix} ${i}`;
		if (!used.has(name)) return name;
	}
}

// --- Block helpers ---

export function createContextBlock(type: string, name: string = ''): ContextBlock {
	return {
		kind: 'context',
		id: createId(),
		name,
		info: { type, tags: {}, description: '' },
		priority: 0,
		content: [{ id: createId(), content_type: 'text', value: '' }],
		enabled: true
	};
}

export function createRawBlock(name: string, mode: RawBlockMode = 'template'): RawBlock {
	return { kind: 'raw', id: createId(), name, text: '', mode };
}

export function createScriptBlock(name: string = ''): ScriptBlock {
	return { kind: 'script', id: createId(), name, text: '' };
}

export function createBlockNode(): BlockNode {
	return { kind: 'block', block: { kind: 'none', id: createId(), name: '' } };
}

export function createContextBlockNode(type: string): BlockNode {
	return { kind: 'block', block: createContextBlock(type) };
}

export function createRawBlockNode(name: string): BlockNode {
	return { kind: 'block', block: createRawBlock(name) };
}

export function createFolderNode(name: string): BlockNode {
	return {
		kind: 'folder',
		folder: { id: createId(), name, enabled: true, children: [] }
	};
}

export function createNode(name: string): Node {
	return {
		id: createId(),
		name,
		kind: 'llm',
		providerId: '',
		model: '',
		temperature: 1.0,
		maxTokens: { input: 16000, output: 4000 },
		selfSpec: { initialValue: '' },
		strategy: { mode: 'once-per-turn' },
		retry: 0,
		assert: 'true',
		messages: [],
		tools: []
	};
}

export function createNodeBlockNode(name: string): BlockNode {
	return { kind: 'node', node: createNode(name) };
}

// --- UI state ---

type NavTarget =
	| { kind: 'bot'; botId: string }
	| { kind: 'settings' };

export type Tab =
	| { kind: 'chat'; sessionId: string }
	| { kind: 'bot-settings'; botId: string }
	| { kind: 'block'; blockId: string; owner: BlockOwner }
	| { kind: 'node'; nodeId: string; owner: BlockOwner }
	| { kind: 'prompt'; promptId: string }
	| { kind: 'profile'; profileId: string }
	| { kind: 'provider'; providerId: string };

export type BlockOwner =
	| { kind: 'bot'; botId: string }
	| { kind: 'prompt'; promptId: string }
	| { kind: 'profile'; profileId: string };

export function tabKey(tab: Tab): string {
	switch (tab.kind) {
		case 'chat': return `chat:${tab.sessionId}`;
		case 'bot-settings': return `bot-settings:${tab.botId}`;
		case 'block': return `block:${tab.blockId}`;
		case 'node': return `node:${tab.nodeId}`;
		case 'prompt': return `prompt:${tab.promptId}`;
		case 'profile': return `profile:${tab.profileId}`;
		case 'provider': return `provider:${tab.providerId}`;
	}
}

class UIState {
	nav = $state<NavTarget>({ kind: 'settings' });
	tabs = $state<Tab[]>([]);
	activeTabIndex = $state(0);

	// Mobile sidebar toggles
	leftSidebarOpen = $state(false);
	rightSidebarOpen = $state(false);

	toggleLeftSidebar() {
		this.leftSidebarOpen = !this.leftSidebarOpen;
		if (this.leftSidebarOpen) this.rightSidebarOpen = false;
	}
	toggleRightSidebar() {
		this.rightSidebarOpen = !this.rightSidebarOpen;
		if (this.rightSidebarOpen) this.leftSidebarOpen = false;
	}
	closeMobileSidebars() {
		this.leftSidebarOpen = false;
		this.rightSidebarOpen = false;
	}

	get activeTab(): Tab | undefined {
		return this.tabs[this.activeTabIndex];
	}

	openTab(tab: Tab) {
		const key = tabKey(tab);
		const existing = this.tabs.findIndex((t) => tabKey(t) === key);
		if (existing >= 0) {
			this.activeTabIndex = existing;
		} else {
			this.tabs = [...this.tabs, tab];
			this.activeTabIndex = this.tabs.length - 1;
		}
	}

	closeTab(index: number) {
		this.tabs = this.tabs.filter((_, i) => i !== index);
		if (this.tabs.length === 0) {
			this.activeTabIndex = 0;
		} else if (this.activeTabIndex >= this.tabs.length) {
			this.activeTabIndex = this.tabs.length - 1;
		} else if (this.activeTabIndex > index) {
			this.activeTabIndex--;
		}
	}

	closeTabsBy(predicate: (tab: Tab) => boolean) {
		const activeTab = this.tabs[this.activeTabIndex];
		this.tabs = this.tabs.filter((t) => !predicate(t));
		if (this.tabs.length === 0) {
			this.activeTabIndex = 0;
		} else if (activeTab && predicate(activeTab)) {
			this.activeTabIndex = Math.min(this.activeTabIndex, this.tabs.length - 1);
		} else if (activeTab) {
			this.activeTabIndex = this.tabs.indexOf(activeTab);
		}
	}

	selectBot(botId: string) {
		this.nav = { kind: 'bot', botId };
		botStore.activeBotId = botId;
		const sessions = sessionStore.forBot(botId);
		if (sessions.length > 0) {
			sessionStore.activeSessionId = sessions[0].id;
			this.openTab({ kind: 'chat', sessionId: sessions[0].id });
		} else {
			sessionStore.activeSessionId = null;
		}
	}

	selectSettings() {
		this.nav = { kind: 'settings' };
	}

	openBotSettings(botId: string) {
		this.openTab({ kind: 'bot-settings', botId });
	}

	openBlock(blockId: string, owner: BlockOwner) {
		this.openTab({ kind: 'block', blockId, owner });
	}

	closeBlock() {
		const idx = this.activeTabIndex;
		if (this.tabs[idx]?.kind === 'block') {
			this.closeTab(idx);
		}
	}

	openNode(nodeId: string, owner: BlockOwner) {
		this.openTab({ kind: 'node', nodeId, owner });
	}

	openPrompt(promptId: string) {
		this.openTab({ kind: 'prompt', promptId });
	}

	openProfile(profileId: string) {
		this.openTab({ kind: 'profile', profileId });
	}

	openProvider(providerId: string) {
		this.openTab({ kind: 'provider', providerId });
	}

	get isSettings(): boolean {
		return this.nav.kind === 'settings';
	}

	get selectedBotId(): string | null {
		return this.nav.kind === 'bot' ? this.nav.botId : null;
	}

	isBlockOpen(blockId: string): boolean {
		const tab = this.activeTab;
		return tab?.kind === 'block' && tab.blockId === blockId;
	}

	isNodeOpen(nodeId: string): boolean {
		const tab = this.activeTab;
		return tab?.kind === 'node' && tab.nodeId === nodeId;
	}

	// --- Coordinated operations (data + UI cleanup) ---

	removeProvider(id: string) {
		providerStore.remove(id);
		this.closeTabsBy((t) => t.kind === 'provider' && t.providerId === id);
	}

	removePrompt(id: string) {
		promptStore.remove(id);
		this.closeTabsBy((t) =>
			(t.kind === 'prompt' && t.promptId === id) ||
			(t.kind === 'node' && t.owner.kind === 'prompt' && t.owner.promptId === id) ||
			(t.kind === 'block' && t.owner.kind === 'prompt' && t.owner.promptId === id)
		);
	}

	removeProfile(id: string) {
		profileStore.remove(id);
		this.closeTabsBy((t) =>
			(t.kind === 'profile' && t.profileId === id) ||
			(t.kind === 'block' && t.owner.kind === 'profile' && t.owner.profileId === id)
		);
	}

	removeBot(id: string) {
		const wasActive = botStore.activeBotId === id;
		const sessionIds = new Set(sessionStore.forBot(id).map((s) => s.id));
		botStore.remove(id);
		this.closeTabsBy((t) =>
			(t.kind === 'bot-settings' && t.botId === id) ||
			(t.kind === 'chat' && sessionIds.has(t.sessionId)) ||
			(t.kind === 'block' && t.owner.kind === 'bot' && t.owner.botId === id) ||
			(t.kind === 'node' && t.owner.kind === 'bot' && t.owner.botId === id)
		);
		if (wasActive) {
			const next = botStore.bots[0];
			if (next) {
				this.selectBot(next.id);
			} else {
				botStore.activeBotId = null;
				this.selectSettings();
			}
		}
	}

	removeSession(id: string) {
		sessionStore.remove(id);
		this.closeTabsBy((t) => t.kind === 'chat' && t.sessionId === id);
	}

	createSession(botId: string, name: string): Session {
		const session = sessionStore.create(botId, name);
		this.openTab({ kind: 'chat', sessionId: session.id });
		return session;
	}

	selectSession(id: string) {
		sessionStore.select(id);
		this.openTab({ kind: 'chat', sessionId: id });
	}

	removeOwnerTreeNode(owner: BlockOwner, treeNodeId: string) {
		const children = getOwnerChildren(owner);
		if (!children) return;
		const found = findTreeNode(children, treeNodeId);
		const removedIds = new Set(found ? collectAllIds([found]) : [treeNodeId]);
		updateOwnerChildren(owner, removeTreeNode(children, treeNodeId));
		this.closeTabsBy((t) =>
			(t.kind === 'block' && removedIds.has(t.blockId)) ||
			(t.kind === 'node' && removedIds.has(t.nodeId))
		);
	}

	// --- Reconciliation (clean stale references after data reload) ---

	reconcile() {
		const nav = this.nav;
		if (nav.kind === 'bot' && !botStore.bots.some((b) => b.id === nav.botId)) {
			const next = botStore.bots[0];
			if (next) {
				this.selectBot(next.id);
			} else {
				botStore.activeBotId = null;
				this.selectSettings();
			}
			return;
		}

		this.closeTabsBy((t) => !this.isTabValid(t));
	}

	private isTabValid(tab: Tab): boolean {
		switch (tab.kind) {
			case 'chat':
				return sessionStore.sessions.some((s) => s.id === tab.sessionId);
			case 'bot-settings':
				return botStore.get(tab.botId) !== undefined;
			case 'block': {
				const children = getOwnerChildren(tab.owner);
				return children !== undefined && findBlock(children, tab.blockId) !== undefined;
			}
			case 'node': {
				const children = getOwnerChildren(tab.owner);
				return children !== undefined && findNodeItem(children, tab.nodeId) !== undefined;
			}
			case 'prompt':
				return promptStore.get(tab.promptId) !== undefined;
			case 'profile':
				return profileStore.get(tab.profileId) !== undefined;
			case 'provider':
				return providerStore.get(tab.providerId) !== undefined;
		}
	}

	// --- Serialization ---

	toJSON() {
		return {
			nav: this.nav,
			tabs: this.tabs,
			activeTabIndex: this.activeTabIndex,
		};
	}

	loadJSON(data: { nav?: NavTarget; tabs?: Tab[]; activeTabIndex?: number }) {
		this.nav = data.nav ?? { kind: 'settings' };
		this.tabs = data.tabs ?? [];
		this.activeTabIndex = data.activeTabIndex ?? 0;
	}
}

export const uiState = new UIState();

// --- Provider store ---

class ProviderStore {
	providers = $state<Provider[]>([]);

	create(name: string): Provider {
		const provider: Provider = {
			id: createId(),
			name,
			api: '',
			endpoint: '',
			apiKey: ''
		};
		this.providers = [...this.providers, provider];
		return provider;
	}

	get(id: string): Provider | undefined {
		return this.providers.find((p) => p.id === id);
	}

	remove(id: string) {
		this.providers = this.providers.filter((p) => p.id !== id);
	}

	update(id: string, updater: (p: Provider) => Provider) {
		this.providers = this.providers.map((p) => (p.id === id ? updater(p) : p));
	}
}

// --- Prompt store ---

class PromptStore {
	prompts = $state<Prompt[]>([]);

	create(name: string): Prompt {
		const prompt: Prompt = {
			id: createId(),
			name,
			children: [],
			contextBindings: [{ name: HISTORY_BINDING_NAME, script: '' }],
			contextParams: []
		};
		this.prompts = [...this.prompts, prompt];
		return prompt;
	}

	get(id: string): Prompt | undefined {
		return this.prompts.find((p) => p.id === id);
	}

	remove(id: string) {
		this.prompts = this.prompts.filter((p) => p.id !== id);
	}

	update(id: string, updater: (p: Prompt) => Prompt) {
		this.prompts = this.prompts.map((p) => (p.id === id ? updater(p) : p));
	}

	addChild(promptId: string, child: BlockNode) {
		this.update(promptId, (p) => ({ ...p, children: addNode(p.children, child) }));
	}

	updateBlock(promptId: string, blockId: string, updater: (b: Block) => Block) {
		this.update(promptId, (p) => ({ ...p, children: treeUpdateBlock(p.children, blockId, updater) }));
	}

	updateNodeItem(promptId: string, nodeId: string, updater: (n: Node) => Node) {
		this.update(promptId, (p) => ({ ...p, children: treeUpdateNodeItem(p.children, nodeId, updater) }));
	}

	import(prompt: Prompt) {
		this.prompts = [...this.prompts, prompt];
	}
}

// --- Profile store ---

class ProfileStore {
	profiles = $state<Profile[]>([]);

	create(name: string): Profile {
		const profile: Profile = {
			id: createId(),
			name,
			children: [],
			contextParams: []
		};
		this.profiles = [...this.profiles, profile];
		return profile;
	}

	get(id: string): Profile | undefined {
		return this.profiles.find((p) => p.id === id);
	}

	remove(id: string) {
		this.profiles = this.profiles.filter((p) => p.id !== id);
	}

	update(id: string, updater: (p: Profile) => Profile) {
		this.profiles = this.profiles.map((p) => (p.id === id ? updater(p) : p));
	}

	addChild(id: string, child: BlockNode) {
		this.update(id, (p) => ({ ...p, children: addNode(p.children, child) }));
	}

	updateBlock(profileId: string, blockId: string, updater: (b: Block) => Block) {
		this.update(profileId, (p) => ({ ...p, children: treeUpdateBlock(p.children, blockId, updater) }));
	}

	updateNodeItem(profileId: string, nodeId: string, updater: (n: Node) => Node) {
		this.update(profileId, (p) => ({ ...p, children: treeUpdateNodeItem(p.children, nodeId, updater) }));
	}

	import(profile: Profile) {
		this.profiles = [...this.profiles, profile];
	}
}

// --- Bot store ---

class BotStore {
	bots = $state<Bot[]>([]);
	activeBotId = $state<string | null>(null);

	get active(): Bot | undefined {
		return this.bots.find((b) => b.id === this.activeBotId);
	}

	get(id: string): Bot | undefined {
		return this.bots.find((b) => b.id === id);
	}

	create(name: string, promptId: string, profileId: string): Bot {
		const bot: Bot = {
			id: createId(),
			name,
			promptId,
			profileId,
			children: [],
			display: { iterator: '', entries: [] },
			regions: [],
			layout: createDefaultLayout(),
			contextParams: []
		};
		this.bots = [...this.bots, bot];
		return bot;
	}

	remove(id: string) {
		this.bots = this.bots.filter((b) => b.id !== id);
		sessionStore.removeForBot(id);
	}

	update(id: string, updater: (b: Bot) => Bot) {
		this.bots = this.bots.map((b) => (b.id === id ? updater(b) : b));
	}

	addChild(botId: string, child: BlockNode) {
		this.update(botId, (b) => ({ ...b, children: addNode(b.children, child) }));
	}

	updateBlock(botId: string, blockId: string, updater: (bl: Block) => Block) {
		this.update(botId, (b) => ({ ...b, children: treeUpdateBlock(b.children, blockId, updater) }));
	}

	updateNodeItem(botId: string, nodeId: string, updater: (n: Node) => Node) {
		this.update(botId, (b) => ({ ...b, children: treeUpdateNodeItem(b.children, nodeId, updater) }));
	}

	import(bot: Bot) {
		this.bots = [...this.bots, bot];
	}
}

// --- Session store ---

class SessionStore {
	sessions = $state<Session[]>([]);
	activeSessionId = $state<string | null>(null);

	get active(): Session | undefined {
		return this.sessions.find((s) => s.id === this.activeSessionId);
	}

	forBot(botId: string): Session[] {
		return this.sessions.filter((s) => s.botId === botId);
	}

	create(botId: string, name: string): Session {
		const session: Session = {
			id: createId(),
			name,
			botId,
			storage: null
		};
		this.sessions = [...this.sessions, session];
		this.activeSessionId = session.id;
		return session;
	}

	remove(id: string) {
		this.sessions = this.sessions.filter((s) => s.id !== id);
		if (this.activeSessionId === id) {
			this.activeSessionId = null;
		}
	}

	removeForBot(botId: string) {
		this.sessions = this.sessions.filter((s) => s.botId !== botId);
		if (this.activeSessionId && !this.sessions.some((s) => s.id === this.activeSessionId)) {
			this.activeSessionId = null;
		}
	}

	update(id: string, updater: (s: Session) => Session) {
		this.sessions = this.sessions.map((s) => (s.id === id ? updater(s) : s));
	}

	select(id: string) {
		this.activeSessionId = id;
	}
}

export const providerStore = new ProviderStore();
export const promptStore = new PromptStore();
export const profileStore = new ProfileStore();
export const botStore = new BotStore();
export const sessionStore = new SessionStore();

// --- HMR: preserve store state across hot reloads ---
if (import.meta.hot) {
	// Restore from previous module's stashed state
	const prev = import.meta.hot.data.storeState as StoreData | undefined;
	if (prev) {
		importData(prev);
	}
	// Stash current state before the module is replaced
	import.meta.hot.dispose(() => {
		import.meta.hot!.data.storeState = exportData();
	});
}

// --- Owner dispatch helpers ---

export function getOwnerChildren(owner: BlockOwner): BlockNode[] | undefined {
	switch (owner.kind) {
		case 'bot': return botStore.get(owner.botId)?.children;
		case 'prompt': return promptStore.get(owner.promptId)?.children;
		case 'profile': return profileStore.get(owner.profileId)?.children;
	}
}

export function updateOwnerBlock(owner: BlockOwner, blockId: string, updater: (b: Block) => Block) {
	switch (owner.kind) {
		case 'bot': botStore.updateBlock(owner.botId, blockId, updater); break;
		case 'prompt': promptStore.updateBlock(owner.promptId, blockId, updater); break;
		case 'profile': profileStore.updateBlock(owner.profileId, blockId, updater); break;
	}
}

export function updateOwnerNodeItem(owner: BlockOwner, nodeId: string, updater: (n: Node) => Node) {
	switch (owner.kind) {
		case 'bot': botStore.updateNodeItem(owner.botId, nodeId, updater); break;
		case 'prompt': promptStore.updateNodeItem(owner.promptId, nodeId, updater); break;
		case 'profile': profileStore.updateNodeItem(owner.profileId, nodeId, updater); break;
	}
}

export function updateOwnerChildren(owner: BlockOwner, children: BlockNode[]) {
	switch (owner.kind) {
		case 'bot': botStore.update(owner.botId, (b) => ({ ...b, children })); break;
		case 'prompt': promptStore.update(owner.promptId, (p) => ({ ...p, children })); break;
		case 'profile': profileStore.update(owner.profileId, (p) => ({ ...p, children })); break;
	}
}

export function addOwnerChild(owner: BlockOwner, child: BlockNode) {
	switch (owner.kind) {
		case 'bot': botStore.addChild(owner.botId, child); break;
		case 'prompt': promptStore.addChild(owner.promptId, child); break;
		case 'profile': profileStore.addChild(owner.profileId, child); break;
	}
}

/** Collect all block names in the resolution scope, excluding `excludeId`. */
export function collectScopeBlockNames(owner: BlockOwner, excludeId: string): string[] {
	const names: string[] = [];
	function gather(children: BlockNode[] | undefined) {
		if (!children) return;
		for (const b of collectBlocks(children)) {
			if (b.id !== excludeId) names.push(b.name);
		}
	}
	switch (owner.kind) {
		case 'prompt':
			gather(promptStore.get(owner.promptId)?.children);
			break;
		case 'profile':
			gather(profileStore.get(owner.profileId)?.children);
			break;
		case 'bot': {
			const bot = botStore.get(owner.botId);
			gather(bot?.children);
			if (bot?.promptId) gather(promptStore.get(bot.promptId)?.children);
			if (bot?.profileId) gather(profileStore.get(bot.profileId)?.children);
			break;
		}
	}
	return names;
}

// --- Data serialization ---

export type StoreData = {
	providers: Provider[];
	prompts: Prompt[];
	profiles: Profile[];
	bots: Bot[];
	sessions: Session[];
};

export function exportData(): StoreData {
	return {
		providers: providerStore.providers,
		prompts: promptStore.prompts,
		profiles: profileStore.profiles,
		bots: botStore.bots,
		sessions: sessionStore.sessions,
	};
}

/** Ensure every Block in stored data has a `kind` discriminant. */
function migrateBlock(block: Record<string, unknown>): Block {
	if ('kind' in block && typeof block.kind === 'string') return block as unknown as Block;
	if ('info' in block) return { ...block, kind: 'context' } as unknown as ContextBlock;
	if ('text' in block && 'mode' in block) return { ...block, kind: 'raw' } as unknown as RawBlock;
	if ('text' in block) return { ...block, kind: 'script' } as unknown as ScriptBlock;
	return { ...block, kind: 'none' } as unknown as NoneBlock;
}

function migrateChildren(children: BlockNode[]): BlockNode[] {
	return children.map((n) => {
		if (n.kind === 'block') return { ...n, block: migrateBlock(n.block as Record<string, unknown>) };
		if (n.kind === 'folder') return { ...n, folder: { ...n.folder, children: migrateChildren(n.folder.children) } };
		return n;
	});
}

export function importData(data: StoreData) {
	providerStore.providers = data.providers;
	promptStore.prompts = (data.prompts ?? []).map((p) => ({ ...p, contextParams: p.contextParams ?? [], children: migrateChildren(p.children ?? []) }));
	profileStore.profiles = (data.profiles ?? []).map((p) => ({ ...p, contextParams: p.contextParams ?? [], children: migrateChildren(p.children ?? []) }));
	botStore.bots = (data.bots ?? []).map((b) => ({ ...b, contextParams: b.contextParams ?? [], children: migrateChildren(b.children ?? []) }));
	sessionStore.sessions = data.sessions ?? [];
	uiState.reconcile();
}

