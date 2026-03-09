/**
 * Svelte action for long-press on touch devices.
 *
 * Sets `data-long-pressed` attribute on the element when a long press is detected.
 * Removes it when the user touches outside the element.
 * Prevents the click event that follows the long-press touchend.
 *
 * Usage: `<div use:longpress={500}>` (duration in ms, default 500)
 *
 * CSS pattern (use :global for runtime attribute):
 *   :global([data-long-pressed] > .action-btn) { opacity: 1; }
 */
export type LongpressOptions = { duration?: number; onlongpress?: () => void };

export function longpress(node: HTMLElement, opts: LongpressOptions | number = 500) {
	let duration = typeof opts === 'number' ? opts : (opts.duration ?? 500);
	let onlongpress = typeof opts === 'number' ? undefined : opts.onlongpress;
	let timer: ReturnType<typeof setTimeout>;
	let preventClick = false;
	let documentListenerActive = false;

	function onTouchStart() {
		preventClick = false;
		timer = setTimeout(() => {
			if (!node.isConnected) return;
			preventClick = true;
			node.setAttribute('data-long-pressed', '');
			node.dispatchEvent(new CustomEvent('longpress', { bubbles: true }));
			onlongpress?.();
			navigator.vibrate?.(30);
			if (!documentListenerActive) {
				documentListenerActive = true;
				document.addEventListener('touchstart', onDocumentTouch);
			}
		}, duration);
	}

	function onTouchMove() {
		clearTimeout(timer);
	}

	function onTouchEnd() {
		clearTimeout(timer);
	}

	function onClick(e: MouseEvent) {
		if (preventClick) {
			e.stopPropagation();
			e.preventDefault();
			preventClick = false;
		}
	}

	function onDocumentTouch(e: TouchEvent) {
		if (!node.contains(e.target as Node)) {
			deactivate();
		}
	}

	function deactivate() {
		node.removeAttribute('data-long-pressed');
		if (documentListenerActive) {
			documentListenerActive = false;
			document.removeEventListener('touchstart', onDocumentTouch);
		}
	}

	node.addEventListener('touchstart', onTouchStart, { passive: true });
	node.addEventListener('touchmove', onTouchMove, { passive: true });
	node.addEventListener('touchend', onTouchEnd, { passive: true });
	node.addEventListener('click', onClick, { capture: true });

	return {
		destroy() {
			clearTimeout(timer);
			deactivate();
			node.removeEventListener('touchstart', onTouchStart);
			node.removeEventListener('touchmove', onTouchMove);
			node.removeEventListener('touchend', onTouchEnd);
			node.removeEventListener('click', onClick, { capture: true });
		},
		update(newOpts: LongpressOptions | number) {
			duration = typeof newOpts === 'number' ? newOpts : (newOpts.duration ?? 500);
			onlongpress = typeof newOpts === 'number' ? undefined : newOpts.onlongpress;
		}
	};
}
