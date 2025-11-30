//! OpenAI Responses API server for llama.cpp.

use std::sync::Arc;

use responses_api::{
    config::Config,
    execution::{Executor, ExecutorConfig},
    mcp::McpClient,
    server::{self, AppState},
    state::InMemoryStore,
};
use tokio::net::TcpListener;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "responses_api=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration
    let config = Config::from_env();
    tracing::info!("Starting Responses API server");
    tracing::info!("LLaMA URL: {}", config.llama_url);
    tracing::info!(
        "MCP servers: {:?}",
        config
            .mcp_servers
            .iter()
            .map(|s| &s.name)
            .collect::<Vec<_>>()
    );

    // Create MCP client and connect to servers
    let mcp_client = McpClient::new(config.mcp_servers.clone());
    if !config.mcp_servers.is_empty() {
        tracing::info!("Connecting to MCP servers...");
        mcp_client.connect_all().await?;
        let tools = mcp_client.available_tools().await;
        tracing::info!("Available MCP tools: {:?}", tools);
    }

    // Create executor (uses a clone of mcp_client)
    let executor_config = ExecutorConfig {
        llama_url: config.llama_url.clone(),
        max_tool_iterations: config.max_tool_iterations,
        timeout_secs: config.timeout.as_secs(),
    };
    let executor = Executor::new(executor_config, mcp_client.clone());

    // Create application state
    let state = Arc::new(AppState {
        executor,
        store: InMemoryStore::new(),
        config: config.clone(),
        mcp: mcp_client,
    });

    // Create router
    let app = server::create_router(state);

    // Start server
    let addr = format!("{}:{}", config.host, config.port);
    tracing::info!("Listening on {}", addr);

    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
