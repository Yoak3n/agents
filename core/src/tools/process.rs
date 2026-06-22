use std::collections::HashMap;
use std::sync::Arc;

use crate::schema::common::{AgentEvent, ProcessStatus};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, mpsc, watch};

/// A managed subprocess with streaming output capture.
struct ManagedProcess {
    child: Arc<Mutex<Child>>,
    status_rx: watch::Receiver<ProcessStatus>,
    output_buffer: Arc<Mutex<Vec<String>>>,
}

/// Manages spawned subprocesses with streaming output and lifecycle control.
///
/// Each spawned process gets a unique ID. Output is streamed line-by-line
/// through `AgentEvent::ProcessOutput` and buffered for later retrieval.
/// All processes are killed on `shutdown()` or `Drop`.
pub struct ProcessManager {
    processes: Mutex<HashMap<String, ManagedProcess>>,
}

impl ProcessManager {
    pub fn new() -> Self {
        Self {
            processes: Mutex::new(HashMap::new()),
        }
    }

    /// Spawn a shell command, returning the process ID immediately.
    ///
    /// stdout and stderr are streamed line-by-line through `event_tx` as
    /// `AgentEvent::ProcessOutput` and buffered for later retrieval via `output()`.
    /// Status changes are reported as `AgentEvent::ProcessStatus`.
    pub async fn spawn(
        &self,
        command: &str,
        event_tx: &mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<String, String> {
        let id = uuid::Uuid::new_v4().to_string();

        let mut child = Self::build_command(command)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("failed to spawn process: {e}"))?;

        let stdout = child.stdout.take().expect("stdout not captured");
        let stderr = child.stderr.take().expect("stderr not captured");

        let child_handle = Arc::new(Mutex::new(child));
        let output_buffer = Arc::new(Mutex::new(Vec::new()));

        let (status_tx, status_rx) = watch::channel(ProcessStatus::Running);

        // Stream stdout in background
        let buf_out = output_buffer.clone();
        let tx_out = event_tx.clone();
        let id_out = id.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                buf_out.lock().await.push(line.clone());
                let _ = tx_out.send(AgentEvent::ProcessOutput {
                    call_id: id_out.clone(),
                    line,
                });
            }
        });

        // Stream stderr in background
        let buf_err = output_buffer.clone();
        let tx_err = event_tx.clone();
        let id_err = id.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                buf_err.lock().await.push(line.clone());
                let _ = tx_err.send(AgentEvent::ProcessOutput {
                    call_id: id_err.clone(),
                    line,
                });
            }
        });

        // Wait for process exit in background
        let child_wait = child_handle.clone();
        let id_exit = id.clone();
        let tx_exit = event_tx.clone();
        tokio::spawn(async move {
            let result = {
                let mut c = child_wait.lock().await;
                c.wait().await
            };
            let status = match result {
                Ok(es) => ProcessStatus::Exited(es.code().unwrap_or(-1)),
                Err(e) => ProcessStatus::Error(e.to_string()),
            };
            let _ = status_tx.send(status.clone());
            let _ = tx_exit.send(AgentEvent::ProcessStatus {
                call_id: id_exit,
                status,
            });
        });

        self.processes.lock().await.insert(
            id.clone(),
            ManagedProcess {
                child: child_handle,
                status_rx,
                output_buffer,
            },
        );

        Ok(id)
    }

    /// Query the current status of a process.
    pub async fn status(&self, id: &str) -> Option<ProcessStatus> {
        let map = self.processes.lock().await;
        map.get(id).map(|p| p.status_rx.borrow().clone())
    }

    /// Get all buffered output lines from a process.
    pub async fn output(&self, id: &str) -> Option<Vec<String>> {
        let map = self.processes.lock().await;
        match map.get(id) {
            Some(p) => Some(p.output_buffer.lock().await.clone()),
            None => None,
        }
    }

    /// Kill a specific process.
    pub async fn kill(&self, id: &str) -> Result<(), String> {
        let map = self.processes.lock().await;
        if let Some(p) = map.get(id) {
            let mut child = p.child.lock().await;
            child
                .start_kill()
                .map_err(|e| format!("failed to kill process: {e}"))?;
            Ok(())
        } else {
            Err(format!("process '{id}' not found"))
        }
    }

    /// Kill all managed processes. Called on shutdown.
    pub async fn kill_all(&self) {
        let mut map = self.processes.lock().await;
        for (id, p) in map.drain() {
            let mut child = p.child.lock().await;
            if let Err(e) = child.start_kill() {
                log::warn!("failed to kill process {id}: {e}");
            }
        }
    }

    /// Remove a finished process from tracking.
    pub async fn remove(&self, id: &str) {
        self.processes.lock().await.remove(id);
    }

    /// Build the shell command appropriate for the current platform.
    fn build_command(command: &str) -> Command {
        #[cfg(windows)]
        {
            let mut cmd = Command::new("cmd");
            cmd.arg("/C").arg(command);
            cmd
        }
        #[cfg(not(windows))]
        {
            let mut cmd = Command::new("sh");
            cmd.arg("-c").arg(command);
            cmd
        }
    }
}

impl Default for ProcessManager {
    fn default() -> Self {
        Self::new()
    }
}
