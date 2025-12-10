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
	// The id param is the server conversation ID (conv_xxx format)
	let conversation = $derived(page.params.id ? conversationStore.get(page.params.id) : undefined);
	let messages = $derived(conversation?.messages ?? []);
	let isStreaming = $state(false);
	let abortController: AbortController | null = $state(null);
	let settingsOpen = $state(false);
	let hasScrolled = $state(false);

	function handleMessageScroll(scrollTop: number) {
		hasScrolled = scrollTop > 0;
	}

	// Track navigation to prevent effect from re-setting activeId during navigation away
	let isNavigatingAway = $state(false);
	beforeNavigate(() => {
		isNavigatingAway = true;
	});

	// Track if we're loading items
	let isLoadingItems = $state(false);

	// Set active and load items if needed
	$effect(() => {
		if (browser && !isNavigatingAway) {
			const convId = page.params.id;

			// Wait for store to finish initial loading before checking if conversation exists
			if (conversationStore.isLoading) {
				logger.debug('ui', 'Waiting for conversations to load', { id: convId });
				return;
			}

			if (!conversation) {
				// Conversation not found - might have expired or server restarted
				logger.warn('ui', 'Conversation not found', { id: convId });
				toastStore.warning('Conversation not found');
				goto('/');
			} else {
				const currentActiveId = untrack(() => conversationStore.activeId);
				if (currentActiveId !== conversation.id) {
					conversationStore.setActive(conversation.id);
				}

				// Load items from server if conversation has no messages
				// (This happens after page refresh - metadata is loaded but not items)
				if (convId && conversation.messages.length === 0 && !isLoadingItems) {
					isLoadingItems = true;
					conversationStore.loadItems(convId).finally(() => {
						isLoadingItems = false;
					});
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

		// conversation.id is now the server ID directly
		const conversationId = conversation.id;

		// Add user message with attachments
		conversationStore.addMessage(conversationId, 'user', text, attachments);

		// Create placeholder for assistant message
		const assistantMessage = conversationStore.addMessage(conversationId, 'assistant', '');
		conversationStore.setMessageStreaming(conversationId, assistantMessage.id, true);

		isStreaming = true;
		abortController = new AbortController();

		// Stream the response
		logger.api.request('POST', '/responses', { conversationId });

		await sendMessageStreaming(
			text,
			{
				model: settingsStore.selectedModel,
				conversationId,
				tools: settingsStore.filterTools([
					{ type: 'web_search' },
					{ type: 'code_interpreter' },
					{ type: 'weather' },
					{ type: 'reader' },
					{ type: 'zimage_turbo' }
				]),
				attachments,
				signal: abortController.signal,
				instructions: settingsStore.systemPrompt || undefined
			},
			{
				onDelta: (content) => {
					conversationStore.updateMessageContent(conversationId, assistantMessage.id, content);
				},
				onOutputItem: (item) => {
					conversationStore.setOutputItem(conversationId, assistantMessage.id, item);
				},
				onFunctionCallArgumentsDelta: (itemId, delta) => {
					conversationStore.updateFunctionCallArguments(
						conversationId,
						assistantMessage.id,
						itemId,
						delta
					);
				},
				onComplete: () => {
					logger.api.streamComplete(
						conversationId,
						conversationStore.get(conversationId)?.messages.find((m) => m.id === assistantMessage.id)
							?.content.length ?? 0
					);
					conversationStore.setMessageStreaming(conversationId, assistantMessage.id, false);
					isStreaming = false;
					abortController = null;
				},
				onError: (error) => {
					// Don't show error toast for user-initiated cancellation
					if (error.message === 'Request was cancelled') {
						logger.info('api', 'Stream cancelled by user', { conversationId });
						conversationStore.setMessageStreaming(conversationId, assistantMessage.id, false);
						isStreaming = false;
						abortController = null;
						return;
					}

					logger.error('api', 'Stream error', {
						conversationId,
						error: error.message
					});
					toastStore.error(error.message);
					conversationStore.updateMessageContent(
						conversationId,
						assistantMessage.id,
						'Sorry, something went wrong. Please try again.'
					);
					conversationStore.setMessageStreaming(conversationId, assistantMessage.id, false);
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

	async function handleRegenerate() {
		if (!conversation || isStreaming) return;

		logger.ui.event('ConversationPage', 'Regenerate response', { conversationId: conversation.id });

		const conversationId = conversation.id;

		// Remove the last assistant message and get the user prompt
		const userText = conversationStore.removeLastAssistantMessage(conversationId);
		if (!userText) {
			toastStore.error('Cannot regenerate: no message to regenerate');
			return;
		}

		// Re-send the message (don't add user message again, just create new assistant message)
		const assistantMessage = conversationStore.addMessage(conversationId, 'assistant', '');
		conversationStore.setMessageStreaming(conversationId, assistantMessage.id, true);

		isStreaming = true;
		abortController = new AbortController();

		sendMessageStreaming(
			userText,
			{
				model: settingsStore.selectedModel,
				conversationId,
				tools: settingsStore.filterTools([
					{ type: 'web_search' },
					{ type: 'code_interpreter' },
					{ type: 'weather' },
					{ type: 'reader' },
					{ type: 'zimage_turbo' }
				]),
				signal: abortController.signal,
				instructions: settingsStore.systemPrompt || undefined
			},
			{
				onDelta: (content) => {
					conversationStore.updateMessageContent(conversationId, assistantMessage.id, content);
				},
				onOutputItem: (item) => {
					conversationStore.setOutputItem(conversationId, assistantMessage.id, item);
				},
				onFunctionCallArgumentsDelta: (itemId, delta) => {
					conversationStore.updateFunctionCallArguments(
						conversationId,
						assistantMessage.id,
						itemId,
						delta
					);
				},
				onComplete: () => {
					logger.api.streamComplete(
						conversationId,
						conversationStore.get(conversationId)?.messages.find((m) => m.id === assistantMessage.id)
							?.content.length ?? 0
					);
					conversationStore.setMessageStreaming(conversationId, assistantMessage.id, false);
					isStreaming = false;
					abortController = null;
				},
				onError: (error) => {
					if (error.message === 'Request was cancelled') {
						logger.info('api', 'Regenerate cancelled by user', { conversationId });
						conversationStore.setMessageStreaming(conversationId, assistantMessage.id, false);
						isStreaming = false;
						abortController = null;
						return;
					}

					logger.error('api', 'Regenerate error', {
						conversationId,
						error: error.message
					});
					toastStore.error(error.message);
					conversationStore.updateMessageContent(
						conversationId,
						assistantMessage.id,
						'Sorry, something went wrong. Please try again.'
					);
					conversationStore.setMessageStreaming(conversationId, assistantMessage.id, false);
					isStreaming = false;
					abortController = null;
				}
			}
		);
	}

	async function handleEdit(messageId: string, newContent: string) {
		if (!conversation || isStreaming) return;

		logger.ui.event('ConversationPage', 'Edit message', {
			conversationId: conversation.id,
			messageId,
			newContentLength: newContent.length
		});

		const conversationId = conversation.id;

		// Update the message content and mark as edited
		conversationStore.updateMessage(conversationId, messageId, newContent);

		// Remove all messages after the edited one
		conversationStore.removeMessagesAfter(conversationId, messageId);

		// Create a new assistant response
		const assistantMessage = conversationStore.addMessage(conversationId, 'assistant', '');
		conversationStore.setMessageStreaming(conversationId, assistantMessage.id, true);

		isStreaming = true;
		abortController = new AbortController();

		// Stream a new response with the edited content
		sendMessageStreaming(
			newContent,
			{
				model: settingsStore.selectedModel,
				conversationId,
				tools: settingsStore.filterTools([
					{ type: 'web_search' },
					{ type: 'code_interpreter' },
					{ type: 'weather' },
					{ type: 'reader' },
					{ type: 'zimage_turbo' }
				]),
				signal: abortController.signal,
				instructions: settingsStore.systemPrompt || undefined
			},
			{
				onDelta: (content) => {
					conversationStore.updateMessageContent(conversationId, assistantMessage.id, content);
				},
				onOutputItem: (item) => {
					conversationStore.setOutputItem(conversationId, assistantMessage.id, item);
				},
				onFunctionCallArgumentsDelta: (itemId, delta) => {
					conversationStore.updateFunctionCallArguments(
						conversationId,
						assistantMessage.id,
						itemId,
						delta
					);
				},
				onComplete: () => {
					logger.api.streamComplete(
						conversationId,
						conversationStore.get(conversationId)?.messages.find((m) => m.id === assistantMessage.id)
							?.content.length ?? 0
					);
					conversationStore.setMessageStreaming(conversationId, assistantMessage.id, false);
					isStreaming = false;
					abortController = null;
				},
				onError: (error) => {
					if (error.message === 'Request was cancelled') {
						logger.info('api', 'Edit regenerate cancelled by user', { conversationId });
						conversationStore.setMessageStreaming(conversationId, assistantMessage.id, false);
						isStreaming = false;
						abortController = null;
						return;
					}

					logger.error('api', 'Edit regenerate error', {
						conversationId,
						error: error.message
					});
					toastStore.error(error.message);
					conversationStore.updateMessageContent(
						conversationId,
						assistantMessage.id,
						'Sorry, something went wrong. Please try again.'
					);
					conversationStore.setMessageStreaming(conversationId, assistantMessage.id, false);
					isStreaming = false;
					abortController = null;
				}
			}
		);
	}

	// Handle Escape key to stop streaming or close settings
	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Escape') {
			if (settingsOpen) {
				settingsOpen = false;
				e.preventDefault();
			} else if (isStreaming) {
				handleStop();
				e.preventDefault();
			}
		}
	}
</script>

<svelte:window onkeydown={handleKeydown} />

<div class="h-14 flex items-center justify-end gap-3 px-4 transition-colors {hasScrolled ? 'border-b' : ''}">
	<ModelSelector />
	<Button variant="ghost" size="icon" onclick={() => (settingsOpen = true)} aria-label="Settings" data-testid="settings-button">
		<Settings class="h-5 w-5" />
	</Button>
</div>
<MessageList {messages} canRegenerate={!isStreaming} onregenerate={handleRegenerate} onedit={handleEdit} onscroll={handleMessageScroll} />
<ChatInput onsubmit={handleSubmit} onstop={handleStop} disabled={isStreaming} streaming={isStreaming} />
<SettingsPanel open={settingsOpen} onclose={() => (settingsOpen = false)} />
