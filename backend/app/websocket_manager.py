"""
WebSocket Manager for Real-time Pipeline Progress Updates
"""

from typing import Dict, List, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
import json
import asyncio
from datetime import datetime
import logging

from .logger import setup_logger

# Setup logging
logger = setup_logger("websocket_manager", "logs/websocket.log")

class ConnectionManager:
    """Manages WebSocket connections and broadcasts"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.pipeline_status: Dict[str, Any] = {
            "is_running": False,
            "pipeline_id": None,
            "current_step": None,
            "progress": 0,
            "total_steps": 0,
            "messages": [],
            "start_time": None,
            "estimated_completion": None
        }
        self.lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        # Send current status to new connection
        await self.send_personal_message(
            json.dumps({"type": "connection", "status": self.pipeline_status}),
            websocket
        )
        logger.info(f"New WebSocket connection. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending message to WebSocket: {e}")
    
    async def broadcast(self, message: str):
        """Broadcast message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send to connection: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
    
    async def update_pipeline_status(self, status: Dict[str, Any]):
        """Update pipeline status and broadcast to all clients"""
        async with self.lock:
            self.pipeline_status.update(status)
            await self.broadcast(json.dumps({
                "type": "pipeline_update",
                "status": self.pipeline_status,
                "timestamp": datetime.now().isoformat()
            }))
    
    async def start_pipeline(self, pipeline_id: str, total_steps: int):
        """Mark pipeline as started"""
        async with self.lock:
            if self.pipeline_status["is_running"]:
                return False  # Pipeline already running
            
            self.pipeline_status = {
                "is_running": True,
                "pipeline_id": pipeline_id,
                "current_step": "Initializing",
                "progress": 0,
                "total_steps": total_steps,
                "messages": ["Pipeline started"],
                "start_time": datetime.now().isoformat(),
                "estimated_completion": None,
                "completed_steps": []
            }
            
            await self.broadcast(json.dumps({
                "type": "pipeline_start",
                "status": self.pipeline_status,
                "timestamp": datetime.now().isoformat()
            }))
            return True
    
    async def update_step(self, step_name: str, step_number: int, message: str = None):
        """Update current pipeline step"""
        async with self.lock:
            if not self.pipeline_status["is_running"]:
                return
            
            self.pipeline_status["current_step"] = step_name
            # Avoid division by zero
            total_steps = self.pipeline_status["total_steps"]
            self.pipeline_status["progress"] = (step_number / total_steps) * 100 if total_steps > 0 else 0
            
            if message:
                self.pipeline_status["messages"].append(f"[{step_name}] {message}")
                # Keep only last 50 messages
                if len(self.pipeline_status["messages"]) > 50:
                    self.pipeline_status["messages"] = self.pipeline_status["messages"][-50:]
            
            # Estimate completion time
            if step_number > 0 and self.pipeline_status["start_time"]:
                elapsed = (datetime.now() - datetime.fromisoformat(self.pipeline_status["start_time"])).total_seconds()
                rate = step_number / elapsed if elapsed > 0 else 0
                remaining = self.pipeline_status["total_steps"] - step_number
                estimated_seconds = remaining / rate if rate > 0 else 0
                self.pipeline_status["estimated_completion"] = (
                    datetime.now().timestamp() + estimated_seconds
                ) if estimated_seconds > 0 else None
            
            await self.broadcast(json.dumps({
                "type": "step_update",
                "status": self.pipeline_status,
                "timestamp": datetime.now().isoformat()
            }))
    
    async def complete_step(self, step_name: str, result: Any = None):
        """Mark a step as completed"""
        async with self.lock:
            if not self.pipeline_status["is_running"]:
                return
            
            if "completed_steps" not in self.pipeline_status:
                self.pipeline_status["completed_steps"] = []
            
            self.pipeline_status["completed_steps"].append({
                "name": step_name,
                "completed_at": datetime.now().isoformat(),
                "result_summary": str(result)[:200] if result else None
            })
            
            self.pipeline_status["messages"].append(f"✅ Completed: {step_name}")
            
            await self.broadcast(json.dumps({
                "type": "step_complete",
                "step": step_name,
                "status": self.pipeline_status,
                "timestamp": datetime.now().isoformat()
            }))
    
    async def end_pipeline(self, success: bool = True, message: str = None):
        """Mark pipeline as ended"""
        async with self.lock:
            self.pipeline_status["is_running"] = False
            self.pipeline_status["current_step"] = "Completed" if success else "Failed"
            self.pipeline_status["progress"] = 100 if success else self.pipeline_status["progress"]
            self.pipeline_status["end_time"] = datetime.now().isoformat()
            
            if message:
                self.pipeline_status["messages"].append(message)
            
            await self.broadcast(json.dumps({
                "type": "pipeline_end",
                "success": success,
                "status": self.pipeline_status,
                "timestamp": datetime.now().isoformat()
            }))
    
    def is_pipeline_running(self) -> bool:
        """Check if a pipeline is currently running"""
        return self.pipeline_status["is_running"]
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return self.pipeline_status.copy()

# Global WebSocket manager instance
ws_manager = ConnectionManager()
