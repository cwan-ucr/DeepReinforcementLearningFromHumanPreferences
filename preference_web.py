from __future__ import annotations

import argparse
import html
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from sumo_rlhf.preference_data import PreferenceLabel, append_preference
from sumo_rlhf.preference_sampling import (
    format_segment_summary,
    sample_matched_pair,
    segment_start_position,
    segment_start_time,
    segment_source,
)
from sumo_rlhf.trajectory_buffer import TrajectoryBuffer
from sumo_rlhf.trajectory_plot import (
    animate_segment_pair,
    plot_segment_pair,
    segment_animation_payload,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Web UI for pairwise trajectory preferences.")
    parser.add_argument("--segments", default="runs/preference_pool.jsonl")
    parser.add_argument("--output", default="runs/preferences.jsonl")
    parser.add_argument("--plot-dir", default="runs/preference_web_plots")
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument(
        "--media",
        choices=["road", "interactive", "animation", "static"],
        default="road",
        help="Show road-view animation, curve animation, animated GIF, or static PNG comparisons.",
    )
    parser.add_argument(
        "--animation-window-seconds",
        type=float,
        default=10.0,
        help="Fixed time-window length shown in animated comparisons.",
    )
    parser.add_argument("--animation-fps", type=int, default=4)
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.5,
        help="Browser playback speed multiplier for road/curve animations.",
    )
    parser.add_argument("--match-position-tol", type=float, default=30.0)
    parser.add_argument("--match-time-tol", type=float, default=20.0)
    parser.add_argument(
        "--match-mode",
        choices=["time", "position", "both", "random"],
        default="time",
        help="How candidate pairs are matched before preference labeling.",
    )
    parser.add_argument(
        "--allow-same-source",
        action="store_true",
        help="Allow pairs from the same source even when different-source pairs exist.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--exit-when-done",
        action="store_true",
        help="Shut down the web server automatically after the requested pairs are labeled.",
    )
    return parser.parse_args()


class PreferenceWebApp:
    def __init__(self, args):
        self.args = args
        self.buffer = TrajectoryBuffer.load_jsonl(args.segments)
        if len(self.buffer.segments) < 2:
            raise RuntimeError("Need at least two trajectory segments.")
        self.current_pair = None
        self.current_plot = None
        self.current_payload = None
        self.current_matched = False
        self.label_count = 0
        self.next_pair()

    def next_pair(self):
        left, right, matched = sample_matched_pair(
            self.buffer.segments,
            position_tol=self.args.match_position_tol,
            time_tol=self.args.match_time_tol,
            match_mode=self.args.match_mode,
            prefer_different_source=not self.args.allow_same_source,
        )
        self.current_pair = (left, right)
        self.current_matched = matched
        self.current_payload = {
            "left": segment_animation_payload(
                left, window_seconds=self.args.animation_window_seconds
            ),
            "right": segment_animation_payload(
                right, window_seconds=self.args.animation_window_seconds
            ),
            "windowSeconds": float(self.args.animation_window_seconds),
            "playbackSpeed": float(self.args.playback_speed),
            "mode": self.args.media,
        }
        if self.args.media in {"road", "interactive"}:
            self.current_plot = None
            return
        suffix = "gif" if self.args.media == "animation" else "png"
        plot_name = f"pair_{self.label_count:04d}_{left.segment_id}_vs_{right.segment_id}.{suffix}"
        self.current_plot = Path(self.args.plot_dir) / plot_name
        if self.args.media == "animation":
            animate_segment_pair(
                left,
                right,
                self.current_plot,
                window_seconds=self.args.animation_window_seconds,
                fps=self.args.animation_fps,
            )
        else:
            plot_segment_pair(left, right, self.current_plot)

    def save_preference(self, preference: int):
        left, right = self.current_pair
        append_preference(
            self.args.output,
            PreferenceLabel(
                left_id=left.segment_id,
                right_id=right.segment_id,
                preference=preference,
            ),
        )
        self.label_count += 1
        if self.label_count < self.args.pairs:
            self.next_pair()

    def done(self) -> bool:
        return self.label_count >= self.args.pairs


def make_handler(app: PreferenceWebApp):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/plot"):
                self._serve_plot()
                return
            self._serve_page()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            form = parse_qs(body)
            preference = form.get("preference", [None])[0]
            if preference in {"0", "1", "2"} and not app.done():
                app.save_preference(int(preference))
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()
            if app.done() and app.args.exit_when_done:
                threading.Thread(target=self.server.shutdown, daemon=True).start()

        def log_message(self, format, *args):
            return

        def _serve_plot(self):
            if app.current_plot is None:
                self.send_response(404)
                self.end_headers()
                return
            data = app.current_plot.read_bytes()
            self.send_response(200)
            content_type = (
                "image/gif" if app.current_plot.suffix.lower() == ".gif" else "image/png"
            )
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _serve_page(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(self._html().encode("utf-8"))

        def _html(self):
            if app.done():
                return f"""
                <!doctype html>
                <html><head><meta charset="utf-8"><title>Preference Labeling Done</title>
                <style>{CSS}</style></head>
                <body><main><h1>标注完成</h1>
                <p>已保存 {app.label_count} 条偏好到 <code>{html.escape(app.args.output)}</code>。</p>
                </main></body></html>
                """

            left, right = app.current_pair
            pos_delta = abs(segment_start_position(left) - segment_start_position(right))
            time_delta = abs(segment_start_time(left) - segment_start_time(right))
            match_status = "matched" if app.current_matched else "closest fallback"
            source_pair = f"{segment_source(left)} vs {segment_source(right)}"
            media_html = (
                '<canvas id="plotCanvas" class="plot-canvas" aria-label="trajectory comparison animation"></canvas>'
                if app.args.media in {"road", "interactive"}
                else f'<img class="plot" src="/plot?i={app.label_count}" alt="trajectory comparison">'
            )
            animation_script = (
                f"<script>const TRAJECTORY_DATA = {json.dumps(app.current_payload)};</script><script>{ANIMATION_JS}</script>"
                if app.args.media in {"road", "interactive"}
                else ""
            )
            return f"""
            <!doctype html>
            <html>
            <head>
              <meta charset="utf-8">
              <title>Trajectory Preference Labeling</title>
              <style>{CSS}</style>
            </head>
            <body>
              <main>
                <header>
                  <h1>轨迹偏好标注</h1>
                  <div class="progress">{app.label_count + 1} / {app.args.pairs}</div>
                </header>
                <section class="meta">
                  <div>{html.escape(format_segment_summary("LEFT", left))}</div>
                  <div>{html.escape(format_segment_summary("RIGHT", right))}</div>
                  <div>pair match: {match_status}, mode={app.args.match_mode}, source={source_pair}, Δpos={pos_delta:.1f}m, Δtime={time_delta:.1f}s</div>
                </section>
                {media_html}
                <form method="post" class="actions">
                  <button name="preference" value="0" class="left" autofocus>左边更好</button>
                  <button name="preference" value="2" class="neutral">差不多</button>
                  <button name="preference" value="1" class="right">右边更好</button>
                </form>
              </main>
              <script>
                document.addEventListener("keydown", (event) => {{
                  const map = {{"1": "0", "2": "1", "3": "2", "0": "2"}};
                  if (map[event.key]) {{
                    const form = document.querySelector("form");
                    const input = document.createElement("input");
                    input.type = "hidden";
                    input.name = "preference";
                    input.value = map[event.key];
                    form.appendChild(input);
                    form.submit();
                  }}
                }});
              </script>
              {animation_script}
            </body>
            </html>
            """

    return Handler


ANIMATION_JS = r"""
const canvas = document.getElementById("plotCanvas");
const ctx = canvas.getContext("2d");
const startTime = performance.now();
const labels = ["distance (m)", "speed (m/s)", "accel cmd", "gap (m)"];

function resizeCanvas() {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.floor(rect.width * ratio);
  canvas.height = Math.floor(rect.height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function finite(value) {
  return Number.isFinite(value);
}

function signalColor(time) {
  return (time % 90) < 45 ? "#2ca02c" : "#d62728";
}

function mapX(panel, segment, time) {
  const [start, end] = segment.window;
  return panel.x + ((time - start) / (end - start)) * panel.w;
}

function mapY(panel, limits, value) {
  const [low, high] = limits;
  return panel.y + panel.h - ((value - low) / (high - low)) * panel.h;
}

function drawAxes(panel, segment, limits, label, isBottom) {
  ctx.strokeStyle = "#d1d5db";
  ctx.lineWidth = 1;
  ctx.strokeRect(panel.x, panel.y, panel.w, panel.h);
  ctx.fillStyle = "#374151";
  ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  ctx.save();
  ctx.translate(panel.x - 42, panel.y + panel.h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText(label, 0, 0);
  ctx.restore();

  ctx.strokeStyle = "rgba(156, 163, 175, 0.25)";
  for (let i = 1; i < 4; i++) {
    const x = panel.x + (panel.w * i) / 4;
    ctx.beginPath();
    ctx.moveTo(x, panel.y);
    ctx.lineTo(x, panel.y + panel.h);
    ctx.stroke();
  }
  for (let i = 1; i < 3; i++) {
    const y = panel.y + (panel.h * i) / 3;
    ctx.beginPath();
    ctx.moveTo(panel.x, y);
    ctx.lineTo(panel.x + panel.w, y);
    ctx.stroke();
  }

  ctx.fillStyle = "#6b7280";
  ctx.textAlign = "left";
  ctx.fillText(limits[1].toFixed(1), panel.x + 4, panel.y + 13);
  ctx.fillText(limits[0].toFixed(1), panel.x + 4, panel.y + panel.h - 5);
  if (isBottom) {
    ctx.textAlign = "center";
    ctx.fillText(segment.window[0].toFixed(1) + "s", panel.x, panel.y + panel.h + 17);
    ctx.fillText(segment.window[1].toFixed(1) + "s", panel.x + panel.w, panel.y + panel.h + 17);
  }
}

function drawSeries(panel, segment, values, limits, currentTime, color, width = 2, dash = []) {
  const times = segment.time;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash);
  ctx.beginPath();
  let drawing = false;
  for (let i = 0; i < times.length; i++) {
    const time = times[i];
    const value = values[i];
    if (time > currentTime || !finite(value)) {
      drawing = false;
      continue;
    }
    const x = mapX(panel, segment, time);
    const y = mapY(panel, limits, value);
    if (!drawing) {
      ctx.moveTo(x, y);
      drawing = true;
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawTraffic(panel, segment, limits, currentTime) {
  ctx.strokeStyle = "rgba(107, 114, 128, 0.55)";
  ctx.lineWidth = 1.2;
  for (const vehicle of segment.traffic) {
    ctx.beginPath();
    let drawing = false;
    for (let i = 0; i < vehicle.time.length; i++) {
      const time = vehicle.time[i];
      const value = vehicle.position[i];
      if (time > currentTime || !finite(value)) {
        drawing = false;
        continue;
      }
      const x = mapX(panel, segment, time);
      const y = mapY(panel, limits, value);
      if (!drawing) {
        ctx.moveTo(x, y);
        drawing = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  }
}

function drawSignals(panel, segment, limits, currentTime) {
  for (const signalPosition of segment.signals) {
    const sampleCount = 22;
    for (let i = 0; i < sampleCount; i++) {
      const time = segment.window[0] + (i / (sampleCount - 1)) * (segment.window[1] - segment.window[0]);
      if (time > currentTime) continue;
      const x = mapX(panel, segment, time);
      const y = mapY(panel, limits, signalPosition);
      ctx.fillStyle = signalColor(time);
      ctx.strokeStyle = "#111827";
      ctx.lineWidth = 0.5;
      ctx.fillRect(x - 4, y - 4, 8, 8);
      ctx.strokeRect(x - 4, y - 4, 8, 8);
    }
  }
}

function drawCurrentLine(panels, segment, currentTime) {
  const x = mapX(panels[0], segment, currentTime);
  ctx.strokeStyle = "rgba(17, 24, 39, 0.7)";
  ctx.lineWidth = 1;
  for (const panel of panels) {
    ctx.beginPath();
    ctx.moveTo(x, panel.y);
    ctx.lineTo(x, panel.y + panel.h);
    ctx.stroke();
  }
}

function drawSegment(segment, x, y, w, h, title, elapsed) {
  ctx.fillStyle = "#111827";
  ctx.font = "18px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(title, x + w / 2, y);

  const panelGap = 18;
  const top = y + 18;
  const panelH = (h - 18 - panelGap * 3 - 22) / 4;
  const panels = [];
  for (let i = 0; i < 4; i++) {
    panels.push({ x: x + 58, y: top + i * (panelH + panelGap), w: w - 72, h: panelH });
  }

  const currentTime = Math.min(segment.window[0] + elapsed, segment.window[1]);
  for (let i = 0; i < 4; i++) {
    drawAxes(panels[i], segment, segment.yLimits[i], labels[i], i === 3);
  }

  drawTraffic(panels[0], segment, segment.yLimits[0], currentTime);
  drawSignals(panels[0], segment, segment.yLimits[0], currentTime);
  drawSeries(panels[0], segment, segment.position, segment.yLimits[0], currentTime, "#1f77b4", 3);
  drawSeries(panels[1], segment, segment.speed, segment.yLimits[1], currentTime, "#17becf", 2.4);
  drawSeries(panels[2], segment, segment.action, segment.yLimits[2], currentTime, "#9467bd", 2.4);
  drawSeries(panels[3], segment, segment.frontGap, segment.yLimits[3], currentTime, "#8c564b", 2.4);
  drawSeries(panels[3], segment, segment.rearGap, segment.yLimits[3], currentTime, "#bcbd22", 2.4);
  drawCurrentLine(panels, segment, currentTime);

  ctx.fillStyle = "#4b5563";
  ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  ctx.textAlign = "right";
  ctx.fillText("elapsed " + elapsed.toFixed(1) + " / " + TRAJECTORY_DATA.windowSeconds.toFixed(1) + "s", x + w - 8, y + 18);
}

function interpolateAt(times, values, time) {
  if (!times || times.length === 0) return null;
  if (time <= times[0]) return finite(values[0]) ? values[0] : null;
  for (let i = 1; i < times.length; i++) {
    if (time <= times[i]) {
      const prevValue = values[i - 1];
      const nextValue = values[i];
      if (!finite(prevValue) || !finite(nextValue)) return finite(nextValue) ? nextValue : null;
      const span = times[i] - times[i - 1];
      const ratio = span <= 0 ? 0 : (time - times[i - 1]) / span;
      return prevValue + ratio * (nextValue - prevValue);
    }
  }
  const last = values[values.length - 1];
  return finite(last) ? last : null;
}

function stepValueAt(times, values, time) {
  if (!times || times.length === 0) return null;
  let lastValue = null;
  for (let i = 0; i < times.length; i++) {
    if (times[i] > time) break;
    if (finite(values[i])) lastValue = values[i];
  }
  return lastValue;
}

function roadLimits(segment) {
  let low = segment.yLimits[0][0];
  let high = segment.yLimits[0][1];
  if (!finite(low) || !finite(high) || high <= low) {
    low = 0;
    high = 350;
  }
  if (high - low < 80) {
    const mid = (low + high) / 2;
    low = mid - 40;
    high = mid + 40;
  }
  return [Math.max(0, low), high];
}

function mapRoadX(panel, limits, position) {
  return panel.x + ((position - limits[0]) / (limits[1] - limits[0])) * panel.w;
}

function drawRoadGrid(panel, limits) {
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(panel.x, panel.y, panel.w, panel.h);
  ctx.strokeStyle = "#d1d5db";
  ctx.lineWidth = 1;
  ctx.strokeRect(panel.x, panel.y, panel.w, panel.h);

  const roadY = panel.y + panel.h * 0.48;
  const roadH = Math.min(92, panel.h * 0.28);
  ctx.fillStyle = "#4b5563";
  ctx.fillRect(panel.x, roadY - roadH / 2, panel.w, roadH);
  ctx.strokeStyle = "#f9fafb";
  ctx.setLineDash([14, 10]);
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(panel.x, roadY);
  ctx.lineTo(panel.x + panel.w, roadY);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "#6b7280";
  ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  ctx.textAlign = "center";
  const tickStep = 50;
  const firstTick = Math.ceil(limits[0] / tickStep) * tickStep;
  for (let pos = firstTick; pos <= limits[1]; pos += tickStep) {
    const x = mapRoadX(panel, limits, pos);
    ctx.strokeStyle = "rgba(17, 24, 39, 0.18)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, roadY + roadH / 2 + 4);
    ctx.lineTo(x, roadY + roadH / 2 + 14);
    ctx.stroke();
    ctx.fillText(pos.toFixed(0) + "m", x, roadY + roadH / 2 + 30);
  }
  return { roadY, roadH };
}

function drawTrafficSignalRoad(panel, limits, segment, currentTime, roadY, roadH) {
  for (const signalPosition of segment.signals) {
    if (signalPosition < limits[0] || signalPosition > limits[1]) continue;
    const x = mapRoadX(panel, limits, signalPosition);
    ctx.strokeStyle = "#111827";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, roadY - roadH / 2 - 24);
    ctx.lineTo(x, roadY + roadH / 2 + 10);
    ctx.stroke();

    ctx.fillStyle = signalColor(currentTime);
    ctx.strokeStyle = "#111827";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(x, roadY - roadH / 2 - 32, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.strokeStyle = "#f9fafb";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(x, roadY - roadH / 2);
    ctx.lineTo(x, roadY + roadH / 2);
    ctx.stroke();
  }
}

function drawSignalPhaseBar(panel, currentTime) {
  const cycle = 90;
  const green = 45;
  const phase = ((currentTime % cycle) + cycle) % cycle;
  const barW = Math.min(240, panel.w * 0.40);
  const barH = 18;
  const x = panel.x + panel.w - barW - 12;
  const y = panel.y + 48;
  const greenW = (green / cycle) * barW;

  ctx.fillStyle = "#2ca02c";
  ctx.fillRect(x, y, greenW, barH);
  ctx.fillStyle = "#d62728";
  ctx.fillRect(x + greenW, y, barW - greenW, barH);
  ctx.strokeStyle = "#111827";
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, barW, barH);

  const markerX = x + (phase / cycle) * barW;
  ctx.strokeStyle = "#111827";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(markerX, y - 6);
  ctx.lineTo(markerX, y + barH + 6);
  ctx.stroke();

  ctx.fillStyle = "#374151";
  ctx.font = "11px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("0", x, y + barH + 18);
  ctx.textAlign = "center";
  ctx.fillText("45", x + greenW, y + barH + 18);
  ctx.textAlign = "right";
  ctx.fillText("90s", x + barW, y + barH + 18);
  ctx.textAlign = "center";
  ctx.fillText("signal phase", x + barW / 2, y - 8);

  const state = phase < green ? "green" : "red";
  ctx.fillStyle = state === "green" ? "#166534" : "#991b1b";
  ctx.textAlign = "right";
  ctx.fillText(state + " " + phase.toFixed(1) + "s", x + barW, y - 8);
}

function mapMiniChartX(chart, segment, time) {
  const [start, end] = segment.window;
  return chart.x + ((time - start) / (end - start)) * chart.w;
}

function mapMiniChartY(track, limits, value) {
  const [low, high] = limits;
  return track.y + track.h - ((value - low) / (high - low)) * track.h;
}

function drawMiniChartSeries(chart, track, segment, values, limits, currentTime, color, stepMode = false, dash = []) {
  const times = segment.time;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.8;
  ctx.setLineDash(dash);
  ctx.beginPath();
  let drawing = false;
  let prevX = null;
  let prevY = null;
  for (let i = 0; i < times.length; i++) {
    const time = times[i];
    const value = values[i];
    if (time > currentTime || !finite(value)) {
      drawing = false;
      continue;
    }
    const x = mapMiniChartX(chart, segment, time);
    const y = mapMiniChartY(track, limits, value);
    if (!drawing) {
      ctx.moveTo(x, y);
      drawing = true;
    } else if (stepMode && prevX !== null && prevY !== null) {
      ctx.lineTo(x, prevY);
      ctx.lineTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
    prevX = x;
    prevY = y;
  }
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawRoadKinematicsChart(panel, segment, currentTime) {
  const chart = {
    x: panel.x + 14,
    y: panel.y + 18,
    w: Math.min(320, panel.w * 0.46),
    h: 142,
  };
  const speedTrack = { x: chart.x + 48, y: chart.y + 24, w: chart.w - 60, h: 42 };
  const accelTrack = { x: chart.x + 48, y: chart.y + 86, w: chart.w - 60, h: 42 };
  const speedLimits = segment.yLimits[1];
  const accelLimits = segment.yLimits[2];

  ctx.fillStyle = "rgba(249, 250, 251, 0.96)";
  ctx.fillRect(chart.x, chart.y, chart.w, chart.h);
  ctx.strokeStyle = "#d1d5db";
  ctx.lineWidth = 1;
  ctx.strokeRect(chart.x, chart.y, chart.w, chart.h);

  ctx.fillStyle = "#374151";
  ctx.font = "11px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("speed / accel", chart.x + chart.w / 2, chart.y + 12);

  for (const track of [speedTrack, accelTrack]) {
    ctx.strokeStyle = "rgba(156, 163, 175, 0.35)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(track.x, track.y + track.h);
    ctx.lineTo(track.x + track.w, track.y + track.h);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(track.x, track.y + track.h / 2);
    ctx.lineTo(track.x + track.w, track.y + track.h / 2);
    ctx.stroke();
  }

  ctx.fillStyle = "#1f77b4";
  ctx.textAlign = "right";
  ctx.fillText("v", speedTrack.x - 28, speedTrack.y + 25);
  ctx.fillStyle = "#9467bd";
  ctx.fillText("a", accelTrack.x - 28, accelTrack.y + 25);
  const isRlSegment = segment.id.startsWith("rl-");

  ctx.fillStyle = "#6b7280";
  ctx.font = "10px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  ctx.textAlign = "right";
  ctx.fillText(speedLimits[1].toFixed(1), speedTrack.x - 4, speedTrack.y + 4);
  ctx.fillText(speedLimits[0].toFixed(1), speedTrack.x - 4, speedTrack.y + speedTrack.h);
  ctx.fillText(accelLimits[1].toFixed(1), accelTrack.x - 4, accelTrack.y + 4);
  ctx.fillText(accelLimits[0].toFixed(1), accelTrack.x - 4, accelTrack.y + accelTrack.h);

  ctx.font = "10px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  ctx.textAlign = "left";
  ctx.fillStyle = "#9467bd";
  ctx.fillText(isRlSegment ? "cmd" : "accel", accelTrack.x + accelTrack.w - 54, accelTrack.y + 10);
  if (isRlSegment) {
    ctx.fillStyle = "#f97316";
    ctx.fillText("actual", accelTrack.x + accelTrack.w - 54, accelTrack.y + 24);
  }

  drawMiniChartSeries(chart, speedTrack, segment, segment.speed, speedLimits, currentTime, "#1f77b4");
  drawMiniChartSeries(chart, accelTrack, segment, segment.action, accelLimits, currentTime, "#9467bd", true);
  if (isRlSegment && segment.actualAccel) {
    drawMiniChartSeries(
      chart,
      accelTrack,
      segment,
      segment.actualAccel,
      accelLimits,
      currentTime,
      "#f97316",
      false,
      [5, 4],
    );
  }

  const markerX = mapMiniChartX(chart, segment, currentTime);
  ctx.strokeStyle = "rgba(17, 24, 39, 0.62)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(markerX, speedTrack.y - 2);
  ctx.lineTo(markerX, accelTrack.y + accelTrack.h + 2);
  ctx.stroke();
}

function drawVehicleRoad(panel, limits, position, y, color, label, isEgo = false) {
  if (!finite(position) || position < limits[0] || position > limits[1]) return;
  const x = mapRoadX(panel, limits, position);
  const length = isEgo ? 34 : 26;
  const height = isEgo ? 18 : 14;
  ctx.fillStyle = color;
  ctx.strokeStyle = isEgo ? "#0f172a" : "#6b7280";
  ctx.lineWidth = isEgo ? 1.8 : 1;
  ctx.beginPath();
  ctx.roundRect(x - length / 2, y - height / 2, length, height, 4);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = "rgba(255, 255, 255, 0.72)";
  ctx.beginPath();
  ctx.moveTo(x + length / 2 - 3, y);
  ctx.lineTo(x + length / 2 - 10, y - 5);
  ctx.lineTo(x + length / 2 - 10, y + 5);
  ctx.closePath();
  ctx.fill();

  if (label) {
    ctx.fillStyle = isEgo ? "#1d4ed8" : "#4b5563";
    ctx.font = isEgo ? "13px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif" : "11px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(label, x, y - height / 2 - 6);
  }
}

function assignVehicleRows(vehiclePositions, panel, limits, egoPosition) {
  const rowOffsets = [-26, -13, 13, 26, -38, 38];
  const occupiedRight = rowOffsets.map(() => -Infinity);
  const minSeparation = 34;
  const assignments = [];
  const sorted = vehiclePositions
    .filter((item) => finite(item.position))
    .map((item) => ({ ...item, x: mapRoadX(panel, limits, item.position) }))
    .sort((a, b) => a.x - b.x);
  const egoX = finite(egoPosition) ? mapRoadX(panel, limits, egoPosition) : null;

  for (const vehicle of sorted) {
    let bestRow = 0;
    let bestScore = Infinity;
    for (let row = 0; row < rowOffsets.length; row++) {
      const separation = vehicle.x - occupiedRight[row];
      const overlapsEgo =
        egoX !== null && Math.abs(vehicle.x - egoX) < minSeparation && Math.abs(rowOffsets[row]) < 18;
      const penalty = separation < minSeparation ? 1000 + (minSeparation - separation) : 0;
      const egoPenalty = overlapsEgo ? 500 : 0;
      const score = penalty + egoPenalty + Math.abs(rowOffsets[row]);
      if (score < bestScore) {
        bestScore = score;
        bestRow = row;
      }
    }
    occupiedRight[bestRow] = vehicle.x + minSeparation;
    assignments.push({ ...vehicle, offset: rowOffsets[bestRow] });
  }
  return assignments;
}

function drawRoadMetrics(segment, panel, currentTime) {
  const speed = interpolateAt(segment.time, segment.speed, currentTime);
  const action = stepValueAt(segment.time, segment.action, currentTime);
  const frontGap = interpolateAt(segment.time, segment.frontGap, currentTime);
  const rearGap = interpolateAt(segment.time, segment.rearGap, currentTime);
  const isRlSegment = segment.id.startsWith("rl-");
  const items = [
    ["speed", speed, "m/s"],
    [isRlSegment ? "cmd accel" : "accel", action, "m/s²"],
    ["front gap", frontGap, "m"],
    ["rear gap", rearGap, "m"],
  ];
  const boxY = panel.y + panel.h - 74;
  const boxW = panel.w / items.length;
  for (let i = 0; i < items.length; i++) {
    const [name, value, unit] = items[i];
    const x = panel.x + i * boxW;
    ctx.fillStyle = "#f9fafb";
    ctx.fillRect(x + 4, boxY, boxW - 8, 54);
    ctx.strokeStyle = "#e5e7eb";
    ctx.strokeRect(x + 4, boxY, boxW - 8, 54);
    ctx.fillStyle = "#6b7280";
    ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(name, x + boxW / 2, boxY + 18);
    ctx.fillStyle = "#111827";
    ctx.font = "15px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
    const text = finite(value) ? value.toFixed(1) + " " + unit : "-";
    ctx.fillText(text, x + boxW / 2, boxY + 40);
  }
}

function drawRoadSegment(segment, x, y, w, h, title, elapsed) {
  const panel = { x: x + 10, y: y + 34, w: w - 20, h: h - 44 };
  const currentTime = Math.min(segment.window[0] + elapsed, segment.window[1]);
  const limits = roadLimits(segment);
  const { roadY, roadH } = drawRoadGrid(panel, limits);

  ctx.fillStyle = "#111827";
  ctx.font = "18px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(title, x + w / 2, y + 18);

  ctx.fillStyle = "#4b5563";
  ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  ctx.textAlign = "right";
  ctx.fillText(
    "SUMO time " + currentTime.toFixed(1) + "s  |  elapsed " + elapsed.toFixed(1) + " / " + TRAJECTORY_DATA.windowSeconds.toFixed(1) + "s",
    x + w - 18,
    y + 37
  );

  drawTrafficSignalRoad(panel, limits, segment, currentTime, roadY, roadH);
  drawRoadKinematicsChart(panel, segment, currentTime);
  drawSignalPhaseBar(panel, currentTime);

  const egoPosition = interpolateAt(segment.time, segment.position, currentTime);
  const trafficVehicles = [];
  for (const vehicle of segment.traffic) {
    const position = interpolateAt(vehicle.time, vehicle.position, currentTime);
    if (finite(position) && position >= limits[0] && position <= limits[1]) {
      trafficVehicles.push({ id: vehicle.id, position });
    }
  }
  for (const vehicle of assignVehicleRows(trafficVehicles, panel, limits, egoPosition)) {
    drawVehicleRoad(panel, limits, vehicle.position, roadY + vehicle.offset, "#9ca3af", "", false);
  }
  drawVehicleRoad(panel, limits, egoPosition, roadY, "#2563eb", "ego", true);

  const egoX = finite(egoPosition) ? mapRoadX(panel, limits, egoPosition) : null;
  if (egoX !== null) {
    ctx.strokeStyle = "rgba(37, 99, 235, 0.22)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(egoX, roadY - roadH / 2 - 18);
    ctx.lineTo(egoX, panel.y + panel.h - 86);
    ctx.stroke();
  }

  drawRoadMetrics(segment, panel, currentTime);
}

function draw() {
  resizeCanvas();
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, rect.width, rect.height);

  const playbackSpeed = TRAJECTORY_DATA.playbackSpeed || 1.0;
  const elapsed = (((performance.now() - startTime) / 1000) * playbackSpeed) % TRAJECTORY_DATA.windowSeconds;
  const gap = 22;
  const colW = (rect.width - gap) / 2;
  if (TRAJECTORY_DATA.mode === "road") {
    drawRoadSegment(TRAJECTORY_DATA.left, 0, 18, colW, rect.height - 28, "LEFT: " + TRAJECTORY_DATA.left.id, elapsed);
    drawRoadSegment(TRAJECTORY_DATA.right, colW + gap, 18, colW, rect.height - 28, "RIGHT: " + TRAJECTORY_DATA.right.id, elapsed);
  } else {
    drawSegment(TRAJECTORY_DATA.left, 0, 24, colW, rect.height - 34, "LEFT: " + TRAJECTORY_DATA.left.id, elapsed);
    drawSegment(TRAJECTORY_DATA.right, colW + gap, 24, colW, rect.height - 34, "RIGHT: " + TRAJECTORY_DATA.right.id, elapsed);
  }
  requestAnimationFrame(draw);
}

window.addEventListener("resize", resizeCanvas);
draw();
"""


CSS = """
body {
  margin: 0;
  background: #f4f6f8;
  color: #111827;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main {
  max-width: 1500px;
  margin: 0 auto;
  padding: 18px 24px 28px;
}
header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}
h1 {
  font-size: 24px;
  margin: 0 0 12px;
}
.progress {
  font-size: 15px;
  color: #4b5563;
}
.meta {
  display: grid;
  gap: 6px;
  margin-bottom: 12px;
  color: #374151;
  font-size: 14px;
}
.plot {
  display: block;
  width: 100%;
  max-height: calc(100vh - 210px);
  object-fit: contain;
  background: white;
  border: 1px solid #d1d5db;
}
.plot-canvas {
  display: block;
  width: 100%;
  height: calc(100vh - 210px);
  min-height: 620px;
  background: white;
  border: 1px solid #d1d5db;
}
.actions {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 12px;
  margin-top: 14px;
}
button {
  appearance: none;
  border: 1px solid #9ca3af;
  background: white;
  color: #111827;
  min-height: 48px;
  font-size: 17px;
  font-weight: 650;
  cursor: pointer;
}
button:hover {
  background: #eef2ff;
  border-color: #6366f1;
}
button.left {
  border-color: #2563eb;
}
button.right {
  border-color: #16a34a;
}
button.neutral {
  border-color: #6b7280;
}
code {
  background: #e5e7eb;
  padding: 2px 5px;
}
"""


def main():
    args = parse_args()
    app = PreferenceWebApp(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    url = f"http://{args.host}:{args.port}"
    print(f"Open {url} to label trajectory preferences.", flush=True)
    print("Keyboard shortcuts: 1=left, 2=right, 3/0=neutral.", flush=True)
    print("Press Ctrl+C in this terminal to stop the server.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
