//! OpenAI Responses API server.

use std::path::PathBuf;
use std::sync::Arc;

use responses_api::{
    config::Config,
    containers::ContainerStore,
    execution::{Executor, ExecutorConfig},
    mcp::McpClient,
    server::{self, AppState},
    storage::{
        StorageConfig, create_conversation_store, create_generic_store, create_response_store,
    },
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

    // Validate required configuration
    if config.models.is_empty() {
        tracing::error!("MODELS_CONFIG must define at least one model");
        std::process::exit(1);
    }

    tracing::info!("Starting Responses API server");
    tracing::info!(
        "Configured models: {:?}",
        config.models.iter().map(|m| &m.id).collect::<Vec<_>>()
    );
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

    // Create container store for code interpreter file outputs
    // Use /data/containers for persistence across restarts (Docker volume mounted)
    let containers = ContainerStore::with_persistence(PathBuf::from("/data/containers"));

    // Create executor (uses a clone of mcp_client and containers)
    let executor_config = ExecutorConfig {
        models: config.models.clone(),
        max_tool_iterations: config.max_tool_iterations,
        timeout_secs: config.timeout.as_secs(),
    };
    let executor = Executor::new(executor_config, mcp_client.clone(), containers.clone())?;

    // Configure storage backend from environment
    let storage_config = StorageConfig::from_env();
    tracing::info!(?storage_config, "Using storage backend");

    // Create application state
    let state = Arc::new(AppState {
        executor,
        store: create_response_store(&storage_config).await,
        conversations: create_conversation_store(&storage_config).await,
        generic_store: create_generic_store(&storage_config).await,
        config: config.clone(),
        mcp: mcp_client,
        containers,
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
