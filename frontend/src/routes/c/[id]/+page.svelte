<script lang="ts">
	import { untrack } from 'svelte';
	import { goto, beforeNavigate } from '$app/navigation';
	import { page } from '$app/state';
	import { browser } from '$app/environment';
	import { ChatInput, MessageList } from '$lib/components/chat';
	import { sendMessageStreaming } from '$lib/api';
	import { conversationStore } from '$lib/stores';

	// Get conversation from store based on URL param
	let conversation = $derived(conversationStore.get(page.params.id));
	let messages = $derived(conversation?.messages ?? []);
	let isStreaming = $state(false);

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
				// Use untrack to prevent this effect from re-running when activeId changes
				const currentActiveId = untrack(() => conversationStore.activeId);
				if (currentActiveId !== conversation.id) {
					conversationStore.setActive(conversation.id);
				}
			}
		}
	});

	async function handleSubmit(text: string) {
		if (!conversation) return;

		// Add user message
		conversationStore.addMessage(conversation.id, 'user', text);

		// Create placeholder for assistant message
		const assistantMessage = conversationStore.addMessage(conversation.id, 'assistant', '');
		conversationStore.setMessageStreaming(conversation.id, assistantMessage.id, true);

		isStreaming = true;

		// Stream the response
		await sendMessageStreaming(
			text,
			{
				previousResponseId: conversation.lastResponseId
			},
			{
				onDelta: (content) => {
					conversationStore.updateMessageContent(conversation!.id, assistantMessage.id, content);
				},
				onComplete: (responseId) => {
					conversationStore.updateLastResponseId(conversation!.id, responseId);
					conversationStore.setMessageStreaming(conversation!.id, assistantMessage.id, false);
					isStreaming = false;
				},
				onError: (error) => {
					console.error('Stream error:', error);
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
