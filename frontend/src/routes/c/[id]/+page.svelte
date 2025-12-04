<script lang="ts">
	import { untrack } from 'svelte';
	import { goto, beforeNavigate } from '$app/navigation';
	import { page } from '$app/state';
	import { browser } from '$app/environment';
	import { Settings } from 'lucide-svelte';
	import { ChatInput, MessageList } from '$lib/components/chat';
	import { ModelSelector, SettingsPanel } from '$lib/components/settings';
	import { Button } from '$lib/components/ui/button';
	import { sendMessageStreaming } from '$lib/api';
	import { conversationStore, settingsStore, toastStore } from '$lib/stores';
	import { logger } from '$lib/utils/logger';
	import type { Attachment } from '$lib/utils/files';

	// Get conversation from store based on URL param
	// The id param is always defined since this is a [id] route
	let conversation = $derived(page.params.id ? conversationStore.get(page.params.id) : undefined);
	let messages = $derived(conversation?.messages ?? []);
	let isStreaming = $state(false);
	let abortController: AbortController | null = $state(null);
	let settingsOpen = $state(false);

	// Track navigation to prevent effect from re-setting activeId during navigation away
	let isNavigatingAway = $state(false);
	beforeNavigate(() => {
		isNavigatingAway = true;
	});

	// Set active and redirect if not found
	$effect(() => {
		if (browser && !isNavigatingAway) {
			if (!conversation) {
				goto('/');
			} else {
				const currentActiveId = untrack(() => conversationStore.activeId);
				if (currentActiveId !== conversation.id) {
					conversationStore.setActive(conversation.id);
				}
			}
		}
	});

	async function handleSubmit(text: string, attachments: Attachment[]) {
		if (!conversation) {
			logger.warn('ui', 'handleSubmit called with no conversation');
			return;
		}

		logger.ui.event('ConversationPage', 'handleSubmit called', {
			conversationId: conversation.id,
			textLength: text.length,
			attachmentCount: attachments.length,
			messageCount: conversation.messages.length
		});

		// Add user message with attachments
		conversationStore.addMessage(conversation.id, 'user', text, attachments);

		// Create placeholder for assistant message
		const assistantMessage = conversationStore.addMessage(conversation.id, 'assistant', '');
		conversationStore.setMessageStreaming(conversation.id, assistantMessage.id, true);

		isStreaming = true;
		abortController = new AbortController();

		// Stream the response
		logger.api.request('POST', '/responses', {
			conversationId: conversation.id,
			previousResponseId: conversation.lastResponseId
		});

		await sendMessageStreaming(
			text,
			{
				model: settingsStore.selectedModel,
				previousResponseId: conversation.lastResponseId,
				tools: settingsStore.filterTools([
					{ type: 'web_search' },
					{ type: 'code_interpreter' },
					{ type: 'weather' },
					{ type: 'reader' }
				]),
				attachments,
				signal: abortController.signal
			},
			{
				onDelta: (content) => {
					conversationStore.updateMessageContent(conversation!.id, assistantMessage.id, content);
				},
				onOutputItem: (item) => {
					conversationStore.setOutputItem(conversation!.id, assistantMessage.id, item);
				},
				onComplete: (responseId) => {
					logger.api.streamComplete(
						conversation!.id,
						conversationStore.get(conversation!.id)?.messages.find((m) => m.id === assistantMessage.id)
							?.content.length ?? 0
					);
					conversationStore.updateLastResponseId(conversation!.id, responseId);
					conversationStore.setMessageStreaming(conversation!.id, assistantMessage.id, false);
					isStreaming = false;
					abortController = null;
				},
				onError: (error) => {
					// Don't show error toast for user-initiated cancellation
					if (error.message === 'Request was cancelled') {
						logger.info('api', 'Stream cancelled by user', { conversationId: conversation!.id });
						conversationStore.setMessageStreaming(conversation!.id, assistantMessage.id, false);
						isStreaming = false;
						abortController = null;
						return;
					}

					logger.error('api', 'Stream error', {
						conversationId: conversation!.id,
						error: error.message
					});
					toastStore.error(error.message);
					conversationStore.updateMessageContent(
						conversation!.id,
						assistantMessage.id,
						'Sorry, something went wrong. Please try again.'
					);
					conversationStore.setMessageStreaming(conversation!.id, assistantMessage.id, false);
					isStreaming = false;
					abortController = null;
				}
			}
		);
	}

	function handleStop() {
		if (abortController) {
			logger.ui.event('ConversationPage', 'Stop streaming', { conversationId: conversation?.id });
			abortController.abort();
		}
	}

	function handleRegenerate() {
		if (!conversation || isStreaming) return;

		logger.ui.event('ConversationPage', 'Regenerate response', { conversationId: conversation.id });

		// Remove the last assistant message and get the user prompt
		const userText = conversationStore.removeLastAssistantMessage(conversation.id);
		if (!userText) {
			toastStore.error('Cannot regenerate: no message to regenerate');
			return;
		}

		// Re-send the message (don't add user message again, just create new assistant message)
		const assistantMessage = conversationStore.addMessage(conversation.id, 'assistant', '');
		conversationStore.setMessageStreaming(conversation.id, assistantMessage.id, true);

		isStreaming = true;
		abortController = new AbortController();

		sendMessageStreaming(
			userText,
			{
				model: settingsStore.selectedModel,
				previousResponseId: conversation.lastResponseId,
				tools: settingsStore.filterTools([
					{ type: 'web_search' },
					{ type: 'code_interpreter' },
					{ type: 'weather' },
					{ type: 'reader' }
				]),
				signal: abortController.signal
			},
			{
				onDelta: (content) => {
					conversationStore.updateMessageContent(conversation!.id, assistantMessage.id, content);
				},
				onOutputItem: (item) => {
					conversationStore.setOutputItem(conversation!.id, assistantMessage.id, item);
				},
				onComplete: (responseId) => {
					logger.api.streamComplete(
						conversation!.id,
						conversationStore.get(conversation!.id)?.messages.find((m) => m.id === assistantMessage.id)
							?.content.length ?? 0
					);
					conversationStore.updateLastResponseId(conversation!.id, responseId);
					conversationStore.setMessageStreaming(conversation!.id, assistantMessage.id, false);
					isStreaming = false;
					abortController = null;
				},
				onError: (error) => {
					if (error.message === 'Request was cancelled') {
						logger.info('api', 'Regenerate cancelled by user', { conversationId: conversation!.id });
						conversationStore.setMessageStreaming(conversation!.id, assistantMessage.id, false);
						isStreaming = false;
						abortController = null;
						return;
					}

					logger.error('api', 'Regenerate error', {
						conversationId: conversation!.id,
						error: error.message
					});
					toastStore.error(error.message);
					conversationStore.updateMessageContent(
						conversation!.id,
						assistantMessage.id,
						'Sorry, something went wrong. Please try again.'
					);
					conversationStore.setMessageStreaming(conversation!.id, assistantMessage.id, false);
					isStreaming = false;
					abortController = null;
				}
			}
		);
	}
</script>

<div class="flex items-center justify-end gap-2 p-2 border-b">
	<ModelSelector />
	<Button variant="ghost" size="icon" onclick={() => (settingsOpen = true)} aria-label="Settings" data-testid="settings-button">
		<Settings class="h-5 w-5" />
	</Button>
</div>
<MessageList {messages} canRegenerate={!isStreaming} onregenerate={handleRegenerate} />
<ChatInput onsubmit={handleSubmit} onstop={handleStop} disabled={isStreaming} streaming={isStreaming} />
<SettingsPanel open={settingsOpen} onclose={() => (settingsOpen = false)} />
