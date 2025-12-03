<script lang="ts">
	import { untrack } from 'svelte';
	import { goto, beforeNavigate } from '$app/navigation';
	import { page } from '$app/state';
	import { browser } from '$app/environment';
	import { ChatInput, MessageList } from '$lib/components/chat';
	import { sendMessageStreaming } from '$lib/api';
	import { conversationStore } from '$lib/stores';
	import { logger } from '$lib/utils/logger';

	// Get conversation from store based on URL param
	// The id param is always defined since this is a [id] route
	let conversation = $derived(page.params.id ? conversationStore.get(page.params.id) : undefined);
	let messages = $derived(conversation?.messages ?? []);
	let isStreaming = $state(false);

	logger.lifecycle.mount('ConversationPage', {
		urlParamId: page.params.id,
		conversationExists: !!conversation,
		activeId: conversationStore.activeId
	});

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

	async function handleSubmit(text: string) {
		if (!conversation) {
			logger.warn('ui', 'handleSubmit called with no conversation');
			return;
		}

		logger.ui.event('ConversationPage', 'handleSubmit called', {
			conversationId: conversation.id,
			textLength: text.length,
			messageCount: conversation.messages.length
		});

		// Add user message
		conversationStore.addMessage(conversation.id, 'user', text);

		// Create placeholder for assistant message
		const assistantMessage = conversationStore.addMessage(conversation.id, 'assistant', '');
		conversationStore.setMessageStreaming(conversation.id, assistantMessage.id, true);

		isStreaming = true;

		// Stream the response
		logger.api.request('POST', '/responses', {
			conversationId: conversation.id,
			previousResponseId: conversation.lastResponseId
		});

		await sendMessageStreaming(
			text,
			{
				previousResponseId: conversation.lastResponseId,
				tools: [{ type: 'web_search' }, { type: 'code_interpreter' }]
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
				},
				onError: (error) => {
					logger.error('api', 'Stream error', {
						conversationId: conversation!.id,
						error: error.message
					});
					conversationStore.updateMessageContent(
						conversation!.id,
						assistantMessage.id,
						`Error: ${error.message}`
					);
					conversationStore.setMessageStreaming(conversation!.id, assistantMessage.id, false);
					isStreaming = false;
				}
			}
		);
	}
</script>

<MessageList {messages} />
<ChatInput onsubmit={handleSubmit} disabled={isStreaming} />
