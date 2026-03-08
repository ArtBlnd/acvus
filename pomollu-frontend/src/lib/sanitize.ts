/** Strip <script> and <style> tags (and their content) from HTML. */
export function sanitizeHtml(html: string): string {
	return html
		.replace(/<script[\s>][\s\S]*?<\/script\s*>/gi, '')
		.replace(/<style[\s>][\s\S]*?<\/style\s*>/gi, '');
}
