<script lang="ts">
	import { goto } from '$app/navigation';
	import { Settings } from 'lucide-svelte';
	import { ChatInput, MessageList } from '$lib/components/chat';
	import { ModelSelector, SettingsPanel } from '$lib/components/settings';
	import { Button } from '$lib/components/ui/button';
	import { sendMessageStreaming } from '$lib/api';
	import { conversationStore, settingsStore, toastStore } from '$lib/stores';
	import { logger } from '$lib/utils/logger';
	import type { Attachment } from '$lib/utils/files';

	let isStreaming = $state(false);
	let abortController: AbortController | null = $state(null);
	let settingsOpen = $state(false);
	let hasScrolled = $state(false);

	function handleMessageScroll(scrollTop: number) {
		hasScrolled = scrollTop > 0;
	}

	// Explicitly track activeId to ensure reactivity when it changes to null
	let activeId = $derived(conversationStore.activeId);
	let activeConversation = $derived(activeId ? conversationStore.get(activeId) : undefined);
	let messages = $derived(activeConversation?.messages ?? []);

	async function handleSubmit(text: string, attachments: Attachment[]) {
		logger.ui.event('HomePage', 'handleSubmit called', {
			textLength: text.length,
			attachmentCount: attachments.length,
			hasActiveConversation: !!activeConversation
		});

		// Create conversation if needed
		let conv = activeConversation;
		if (!conv) {
			logger.info('ui', 'Creating new conversation for message');
			conv = conversationStore.create();
		}

		// Add user message with attachments
		conversationStore.addMessage(conv.id, 'user', text, attachments);

		// Create placeholder for assistant message
		const assistantMessage = conversationStore.addMessage(conv.id, 'assistant', '');
		conversationStore.setMessageStreaming(conv.id, assistantMessage.id, true);

		isStreaming = true;
		abortController = new AbortController();

		// Navigate to the conversation URL
		logger.nav.navigate('/', `/c/${conv.id}`, { conversationId: conv.id });
		goto(`/c/${conv.id}`);

		// Stream the response
		logger.api.request('POST', '/responses', { previousResponseId: conv.lastResponseId });

		await sendMessageStreaming(
			text,
			{
				model: settingsStore.selectedModel,
				previousResponseId: conv.lastResponseId,
				tools: settingsStore.filterTools([
					{ type: 'web_search' },
					{ type: 'code_interpreter' },
					{ type: 'weather' },
					{ type: 'reader' }
				]),
				attachments,
				signal: abortController.signal,
				instructions: settingsStore.systemPrompt || undefined
			},
			{
				onDelta: (content) => {
					conversationStore.updateMessageContent(conv!.id, assistantMessage.id, content);
				},
				onOutputItem: (item) => {
					conversationStore.setOutputItem(conv!.id, assistantMessage.id, item);
				},
				onComplete: (responseId) => {
					logger.api.streamComplete(conv!.id, conversationStore.get(conv!.id)?.messages.find(m => m.id === assistantMessage.id)?.content.length ?? 0);
					conversationStore.updateLastResponseId(conv!.id, responseId);
					conversationStore.setMessageStreaming(conv!.id, assistantMessage.id, false);
					isStreaming = false;
					abortController = null;
				},
				onError: (error) => {
					// Don't show error toast for user-initiated cancellation
					if (error.message === 'Request was cancelled') {
						logger.info('api', 'Stream cancelled by user', { conversationId: conv!.id });
						conversationStore.setMessageStreaming(conv!.id, assistantMessage.id, false);
						isStreaming = false;
						abortController = null;
						return;
					}

					logger.error('api', 'Stream error', { conversationId: conv!.id, error: error.message });
					toastStore.error(error.message);
					conversationStore.updateMessageContent(
						conv!.id,
						assistantMessage.id,
						'Sorry, something went wrong. Please try again.'
					);
					conversationStore.setMessageStreaming(conv!.id, assistantMessage.id, false);
					isStreaming = false;
					abortController = null;
				}
			}
		);
	}

	function handleStop() {
		if (abortController) {
			logger.ui.event('HomePage', 'Stop streaming', {});
			abortController.abort();
		}
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
<MessageList {messages} onscroll={handleMessageScroll} />
<ChatInput onsubmit={handleSubmit} onstop={handleStop} disabled={isStreaming} streaming={isStreaming} />
<SettingsPanel open={settingsOpen} onclose={() => (settingsOpen = false)} />
