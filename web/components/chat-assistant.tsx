"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkBreaks from "remark-breaks";
import remarkGfm from "remark-gfm";
import { Mic, Pause, Play, Volume2 } from "lucide-react";

import { Button } from "@/components/ui/button";

type Source = {
  title: string;
  filename: string;
};

type Message = {
  id: number;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
};

const starterQuestion = "What should I do if my meter appears to be faulty?";
const transcriptionSampleRate = 16_000;

async function recordingToWav(recording: Blob): Promise<Blob> {
  const audioContext = new AudioContext();
  try {
    const decodedAudio = await audioContext.decodeAudioData(await recording.arrayBuffer());
    const frameCount = Math.ceil(decodedAudio.duration * transcriptionSampleRate);
    const offlineContext = new OfflineAudioContext(1, frameCount, transcriptionSampleRate);
    const source = offlineContext.createBufferSource();
    source.buffer = decodedAudio;
    source.connect(offlineContext.destination);
    source.start();
    const resampledAudio = await offlineContext.startRendering();
    const samples = resampledAudio.getChannelData(0);
    const wav = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(wav);
    const writeText = (offset: number, value: string) => {
      for (let index = 0; index < value.length; index += 1) {
        view.setUint8(offset + index, value.charCodeAt(index));
      }
    };

    writeText(0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    writeText(8, "WAVE");
    writeText(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, transcriptionSampleRate, true);
    view.setUint32(28, transcriptionSampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeText(36, "data");
    view.setUint32(40, samples.length * 2, true);
    samples.forEach((sample, index) => {
      const clipped = Math.max(-1, Math.min(1, sample));
      view.setInt16(44 + index * 2, clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff, true);
    });

    return new Blob([wav], { type: "audio/wav" });
  } finally {
    await audioContext.close();
  }
}

function normalizeMarkdown(text: string) {
  const slash = String.fromCharCode(92);
  return text
    .replaceAll(`${slash}r${slash}n`, "\n")
    .replaceAll(`${slash}n`, "\n")
    .replaceAll(`${slash}r`, "\n");
}

function isChatResponse(value: unknown): value is { answer: string; sources: Source[] } {
  if (!value || typeof value !== "object") return false;
  const response = value as Record<string, unknown>;
  return (
    typeof response.answer === "string" &&
    Array.isArray(response.sources) &&
    response.sources.every(
      (source) =>
        source &&
        typeof source === "object" &&
        typeof (source as Record<string, unknown>).title === "string" &&
        typeof (source as Record<string, unknown>).filename === "string",
    )
  );
}

export function ChatAssistant() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [isListening, setIsListening] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ttsLoadingId, setTtsLoadingId] = useState<number | null>(null);
  const [ttsPlayingId, setTtsPlayingId] = useState<number | null>(null);
  const [ttsPausedId, setTtsPausedId] = useState<number | null>(null);
  const [ttsNeedsGestureId, setTtsNeedsGestureId] = useState<number | null>(null);
  const [ttsError, setTtsError] = useState<{ id: number; message: string } | null>(null);
  const nextId = useRef(0);
  const sending = useRef(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const playbackContextRef = useRef<AudioContext | null>(null);
  const answerAudioBuffersRef = useRef<Map<number, AudioBuffer>>(new Map());
  const answerAudioSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const activeAudioMessageIdRef = useRef<number | null>(null);
  const playbackStartedAtRef = useRef(0);
  const pausedOffsetRef = useRef(0);
  const ttsLoadingRef = useRef(false);
  const ttsAbortControllerRef = useRef<AbortController | null>(null);
  const ttsRequestIdRef = useRef(0);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isLoading, error]);

  useEffect(() => {
    return () => {
      const recorder = mediaRecorderRef.current;
      if (recorder && recorder.state !== "inactive") {
        recorder.onstop = null;
        recorder.stop();
      }
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
      try {
        answerAudioSourceRef.current?.stop();
      } catch {
        // The source can finish naturally while the component is unmounting.
      }
      answerAudioSourceRef.current?.disconnect();
      answerAudioSourceRef.current = null;
      void playbackContextRef.current?.close();
      playbackContextRef.current = null;
      answerAudioBuffersRef.current.clear();
    };
  }, []);

  function getPlaybackContext() {
    if (!playbackContextRef.current || playbackContextRef.current.state === "closed") {
      playbackContextRef.current = new AudioContext();
    }
    return playbackContextRef.current;
  }

  function stopActiveAnswerPlayback(resetPosition = true) {
    const source = answerAudioSourceRef.current;
    answerAudioSourceRef.current = null;
    activeAudioMessageIdRef.current = null;
    if (source) {
      source.onended = null;
      try {
        source.stop();
      } catch {
        // The source may have ended just before stop() was called.
      }
    }
    if (resetPosition) pausedOffsetRef.current = 0;
    setTtsPlayingId(null);
    setTtsPausedId(null);
  }

  async function startBufferedAnswer(messageId: number, offset: number) {
    if (
      activeAudioMessageIdRef.current !== null &&
      activeAudioMessageIdRef.current !== messageId
    ) {
      stopActiveAnswerPlayback();
    }
    const context = playbackContextRef.current;
    const buffer = answerAudioBuffersRef.current.get(messageId);
    if (!context || !buffer) {
      setTtsNeedsGestureId(messageId);
      setTtsError({ id: messageId, message: "Tap Play to hear the answer." });
      return;
    }

    try {
      if (context.state !== "running") await context.resume();
      if (context.state !== "running") throw new Error("AudioContext did not resume");

      const startOffset = offset >= buffer.duration ? 0 : Math.max(0, offset);
      const source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(context.destination);
      source.onended = () => {
        if (answerAudioSourceRef.current !== source) return;
        answerAudioSourceRef.current = null;
        activeAudioMessageIdRef.current = null;
        pausedOffsetRef.current = 0;
        setTtsPlayingId(null);
        setTtsPausedId(null);
        setTtsNeedsGestureId(null);
      };
      answerAudioSourceRef.current = source;
      activeAudioMessageIdRef.current = messageId;
      playbackStartedAtRef.current = context.currentTime;
      pausedOffsetRef.current = startOffset;
      source.start(0, startOffset);
      setTtsPlayingId(messageId);
      setTtsPausedId(null);
      setTtsNeedsGestureId(null);
      setTtsError(null);
    } catch (playbackError) {
      console.error("[TTS] Web Audio playback could not start", playbackError);
      const failedSource = answerAudioSourceRef.current;
      answerAudioSourceRef.current = null;
      activeAudioMessageIdRef.current = null;
      failedSource?.disconnect();
      try {
        failedSource?.stop();
      } catch {
        // A source that failed during start may already be stopped.
      }
      setTtsNeedsGestureId(messageId);
      setTtsPausedId(null);
      setTtsError({ id: messageId, message: "Tap Play to hear the answer." });
    }
  }

  async function toggleAnswerAudio(message: Message, autoPlay = false) {
    const messageId = message.id;
    const context = playbackContextRef.current;

    // A decoded buffer is reused for replay and the autoplay fallback.
    if (activeAudioMessageIdRef.current === messageId && answerAudioSourceRef.current && context) {
      const buffer = answerAudioBuffersRef.current.get(messageId);
      if (buffer) {
        pausedOffsetRef.current = Math.min(
          buffer.duration,
          pausedOffsetRef.current + context.currentTime - playbackStartedAtRef.current,
        );
      }
      stopActiveAnswerPlayback(false);
      setTtsPausedId(messageId);
      return;
    }

    if (
      activeAudioMessageIdRef.current !== null &&
      activeAudioMessageIdRef.current !== messageId
    ) {
      stopActiveAnswerPlayback();
    }

    if (answerAudioBuffersRef.current.has(messageId)) {
      await startBufferedAnswer(
        messageId,
        ttsPausedId === messageId || ttsNeedsGestureId === messageId
          ? pausedOffsetRef.current
          : 0,
      );
      return;
    }

    if (ttsLoadingRef.current && !autoPlay) return;

    // A new voice answer takes precedence over any older audio generation.
    ttsAbortControllerRef.current?.abort();
    const requestId = ++ttsRequestIdRef.current;
    const abortController = new AbortController();
    ttsAbortControllerRef.current = abortController;

    // Stop the previous answer before fetching or playing another one.
    stopActiveAnswerPlayback();
    setTtsNeedsGestureId(null);
    ttsLoadingRef.current = true;

    setTtsLoadingId(messageId);
    setTtsError(null);
    try {
      if (autoPlay) console.info("[VOICE] requesting TTS");
      const response = await fetch("/api/text-to-speech", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: message.content }),
        signal: abortController.signal,
      });
      if (!response.ok || !response.headers.get("content-type")?.startsWith("audio/wav")) {
        throw new Error("Audio generation failed");
      }

      const audioBytes = await response.arrayBuffer();
      if (requestId !== ttsRequestIdRef.current || abortController.signal.aborted) return;
      const audioContext = getPlaybackContext();
      const audioBuffer = await audioContext.decodeAudioData(audioBytes);
      if (requestId !== ttsRequestIdRef.current || abortController.signal.aborted) return;
      answerAudioBuffersRef.current.set(messageId, audioBuffer);
      if (autoPlay) console.info("[VOICE] TTS audio received");
      if (autoPlay) console.info("[VOICE] starting automatic playback");
      await startBufferedAnswer(messageId, 0);
    } catch (ttsFailure) {
      if (requestId !== ttsRequestIdRef.current || abortController.signal.aborted) return;
      console.error("[TTS] Voice answer audio failed", ttsFailure);
      setTtsNeedsGestureId(answerAudioBuffersRef.current.has(messageId) ? messageId : null);
      setTtsPlayingId(null);
      setTtsError({ id: message.id, message: "Audio playback isn’t available right now. Please try again." });
    } finally {
      if (requestId === ttsRequestIdRef.current) {
        ttsLoadingRef.current = false;
        ttsAbortControllerRef.current = null;
        setTtsLoadingId(null);
      }
    }
  }

  async function transcribeRecording(recording: Blob) {
    setIsTranscribing(true);
    setError(null);
    try {
      const wavRecording = await recordingToWav(recording);
      const response = await fetch("/api/speech-to-text", {
        method: "POST",
        headers: { "Content-Type": "audio/wav" },
        body: wavRecording,
      });
      if (!response.ok) throw new Error("Transcription request failed");

      const data: unknown = await response.json();
      if (!data || typeof data !== "object" || typeof (data as { text?: unknown }).text !== "string") {
        throw new Error("Unexpected transcription response");
      }

      const transcription = (data as { text: string }).text.trim();
      if (!transcription) {
        setError("No speech was detected. Please try recording again.");
        return;
      }
      console.info("[VOICE] transcription received");
      // Reuse the same submission path as typed questions so voice input gets
      // identical chat, loading, and error handling without a second submit.
      await sendQuestion(transcription, "voice");
    } catch {
      setError("We couldn’t transcribe that recording. Please try again.");
    } finally {
      setIsTranscribing(false);
    }
  }

  async function startRecording() {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setError("Voice recording is not supported by this browser.");
      return;
    }

    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      audioChunksRef.current = [];
      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data);
      };
      recorder.onerror = () => {
        stream.getTracks().forEach((track) => track.stop());
        mediaStreamRef.current = null;
        mediaRecorderRef.current = null;
        setIsListening(false);
        setError("Recording stopped unexpectedly. Please try again.");
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        mediaStreamRef.current = null;
        mediaRecorderRef.current = null;
        setIsListening(false);
        const recording = new Blob(audioChunksRef.current, {
          type: recorder.mimeType || "audio/webm",
        });
        audioChunksRef.current = [];
        if (recording.size > 0) {
          void transcribeRecording(recording);
        } else {
          setError("No audio was recorded. Please try again.");
        }
      };
      recorder.start();
      setIsListening(true);
    } catch (recordingError) {
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
      const permissionDenied =
        recordingError instanceof DOMException &&
        (recordingError.name === "NotAllowedError" || recordingError.name === "SecurityError");
      setError(
        permissionDenied
          ? "Microphone access was denied. Allow microphone access in your browser and try again."
          : "We couldn’t start microphone recording. Please try again.",
      );
    }
  }

  function handleMicrophoneClick() {
    const recorder = mediaRecorderRef.current;
    if (recorder?.state === "recording") {
      recorder.stop();
      return;
    }
    console.info("[VOICE] microphone gesture received");

    stopActiveAnswerPlayback();
    if (ttsLoadingRef.current) {
      ttsAbortControllerRef.current?.abort();
      ttsRequestIdRef.current += 1;
      ttsLoadingRef.current = false;
      ttsAbortControllerRef.current = null;
      setTtsLoadingId(null);
    }

    try {
      const audioContext = getPlaybackContext();
      const resumePromise = audioContext.resume();
      void resumePromise.then(() => {
        console.info("[VOICE] AudioContext resumed");
      }).catch((activationError: unknown) => {
        console.error("[VOICE] Could not activate answer audio", activationError);
      });
    } catch (activationError) {
      console.error("[VOICE] Could not create answer AudioContext", activationError);
    }
    void startRecording();
  }

  function handleAnswerAudioClick(message: Message) {
    try {
      const audioContext = getPlaybackContext();
      const resumePromise = audioContext.resume();
      void resumePromise.then(() => toggleAnswerAudio(message)).catch((activationError: unknown) => {
        console.error("[TTS] AudioContext resume failed", activationError);
        setTtsNeedsGestureId(message.id);
        setTtsPausedId(null);
        setTtsError({ id: message.id, message: "Tap Play to hear the answer." });
      });
    } catch (activationError) {
      console.error("[TTS] AudioContext initialization failed", activationError);
      setTtsNeedsGestureId(message.id);
      setTtsPausedId(null);
      setTtsError({ id: message.id, message: "Tap Play to hear the answer." });
    }
  }

  async function sendQuestion(value = question, source: "typed" | "voice" = "typed") {
    const cleanQuestion = value.trim();
    if (!cleanQuestion || sending.current) return;

    sending.current = true;
    setIsLoading(true);
    setError(null);
    setQuestion("");
    setMessages((current) => [
      ...current,
      { id: nextId.current++, role: "user", content: cleanQuestion },
    ]);

    try {
      if (source === "voice") console.info("[VOICE] submitting to /api/chat");
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: cleanQuestion }),
      });

      if (!response.ok) throw new Error("Chat request failed");
      const data: unknown = await response.json();
      if (!isChatResponse(data) || !data.answer.trim()) {
        throw new Error("Unexpected chat response");
      }
      if (source === "voice") console.info("[VOICE] chat response received");

      const assistantMessage: Message = {
        id: nextId.current++,
        role: "assistant",
        content: normalizeMarkdown(data.answer),
        sources: data.sources,
      };
      setMessages((current) => [...current, assistantMessage]);
      if (source === "voice") void toggleAnswerAudio(assistantMessage, true);
    } catch {
      setError(
        "We couldn’t reach the knowledge assistant. Please try again in a moment.",
      );
      setQuestion(cleanQuestion);
    } finally {
      sending.current = false;
      setIsLoading(false);
      inputRef.current?.focus();
    }
  }

  function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void sendQuestion();
  }

  function handleInputKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendQuestion();
    }
  }

  return (
    <main className="min-h-screen bg-[#f4f7f6] px-4 pb-6 pt-5 text-slate-900 sm:px-6 sm:pt-8">
      <div className="mx-auto flex min-h-[calc(100vh-44px)] max-w-5xl flex-col sm:min-h-[calc(100vh-64px)]">
        <header className="mb-5 flex items-center justify-between gap-4 sm:mb-7">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex size-11 shrink-0 items-center justify-center rounded-2xl bg-emerald-800 text-white shadow-sm shadow-emerald-950/15">
              <svg viewBox="0 0 24 24" fill="none" className="size-6" aria-hidden="true">
                <path
                  d="M13.4 2.8 5.8 13h5l-.3 8.2L18.2 11h-5.1l.3-8.2Z"
                  fill="currentColor"
                />
              </svg>
            </div>
            <div className="min-w-0">
              <h1 className="truncate text-base font-semibold tracking-tight sm:text-lg">
                Utility Knowledge Assistant
              </h1>
              <p className="mt-0.5 truncate text-[11px] text-slate-500 sm:text-sm">
                Clear guidance, grounded in approved organizational knowledge.
              </p>
            </div>
          </div>
          <div className="hidden items-center gap-2 rounded-full border border-emerald-100 bg-white px-3 py-1.5 text-xs font-medium text-emerald-800 shadow-sm sm:flex">
            <span className="size-2 rounded-full bg-emerald-500" />
            Knowledge assistant
          </div>
        </header>

        <section className="flex min-h-[600px] flex-1 flex-col overflow-hidden rounded-3xl border border-slate-200/80 bg-white shadow-[0_18px_60px_-35px_rgba(15,23,42,0.28)] sm:min-h-0">
          <div className="flex items-center justify-between border-b border-slate-100 px-5 py-4 sm:px-7">
            <div>
              <p className="text-sm font-semibold text-slate-800">Customer support</p>
              <p className="mt-0.5 text-xs text-slate-500">
                Answers are based on approved utility guidance
              </p>
            </div>
            <div className="flex items-center gap-2 rounded-full bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-800">
              <svg viewBox="0 0 20 20" fill="none" className="size-4" aria-hidden="true">
                <path
                  d="M10 2.5 16 5v4.8c0 3.6-2.5 6.1-6 7.7-3.5-1.6-6-4.1-6-7.7V5l6-2.5Z"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinejoin="round"
                />
                <path d="m7.3 9.9 1.8 1.8 3.7-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Approved knowledge
            </div>
          </div>

          <div
            className="flex-1 overflow-y-auto px-4 py-6 sm:px-8 sm:py-8"
            role="log"
            aria-live="polite"
            aria-label="Conversation"
          >
            {messages.length === 0 ? (
              <div className="mx-auto flex min-h-[390px] max-w-xl flex-col items-center justify-center text-center sm:min-h-[430px]">
                <div className="mb-5 flex size-16 items-center justify-center rounded-3xl bg-emerald-50 text-emerald-800 ring-1 ring-inset ring-emerald-100">
                  <svg viewBox="0 0 24 24" fill="none" className="size-8" aria-hidden="true">
                    <path d="M5 5.75A2.75 2.75 0 0 1 7.75 3h8.5A2.75 2.75 0 0 1 19 5.75v7.5A2.75 2.75 0 0 1 16.25 16h-5.5L6 20v-4.2a2.74 2.74 0 0 1-1-2.05v-8Z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
                    <path d="M8 8h8M8 11.5h5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                  </svg>
                </div>
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-800">
                  Here to help
                </p>
                <h2 className="mt-3 text-2xl font-semibold tracking-tight text-slate-900 sm:text-3xl">
                  What can we help you with?
                </h2>
                <p className="mt-3 max-w-md text-sm leading-6 text-slate-500 sm:text-base">
                  Ask about utility services, procedures, or customer support. Answers are drawn from approved organizational knowledge.
                </p>
                <Button
                  variant="outline"
                  className="mt-7 h-auto max-w-full whitespace-normal rounded-xl border-slate-200 px-4 py-3 text-left font-medium text-slate-700 hover:border-emerald-200 hover:bg-emerald-50 hover:text-emerald-900"
                  onClick={() => void sendQuestion(starterQuestion)}
                  disabled={isLoading || isListening || isTranscribing}
                >
                  <span className="mr-2 text-emerald-700" aria-hidden="true">↳</span>
                  {starterQuestion}
                </Button>
                <p className="mt-3 text-xs text-slate-400">Try this example question</p>
              </div>
            ) : (
              <div className="mx-auto flex max-w-3xl flex-col gap-7">
                {messages.map((message) => (
                  <article
                    key={message.id}
                    className={`flex gap-3 sm:gap-4 ${message.role === "user" ? "justify-end" : "justify-start"}`}
                  >
                    {message.role === "assistant" && (
                      <div className="mt-1 flex size-9 shrink-0 items-center justify-center rounded-xl bg-emerald-800 text-white" aria-hidden="true">
                        <svg viewBox="0 0 24 24" fill="none" className="size-5">
                          <path d="M13.4 2.8 5.8 13h5l-.3 8.2L18.2 11h-5.1l.3-8.2Z" fill="currentColor" />
                        </svg>
                      </div>
                    )}
                    <div className={`min-w-0 max-w-[88%] sm:max-w-[78%] ${message.role === "user" ? "rounded-2xl rounded-tr-md bg-emerald-800 px-4 py-3 text-white shadow-sm sm:px-5" : "pt-1"}`}>
                      {message.role === "user" ? (
                        <p className="whitespace-pre-wrap break-words text-sm leading-6">{message.content}</p>
                      ) : (
                        <>
                          <div className="chat-markdown">
                            <ReactMarkdown
                              remarkPlugins={[remarkGfm, remarkBreaks]}
                              components={{
                                a: ({ href, children }) => (
                                  <a href={href} target="_blank" rel="noreferrer">
                                    {children}
                                  </a>
                                ),
                              }}
                            >
                              {message.content}
                            </ReactMarkdown>
                          </div>
                          {message.sources && message.sources.length > 0 && (
                            <div className="mt-5 border-t border-slate-100 pt-3">
                              <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-400">
                                Sources
                              </p>
                              <ul className="flex flex-wrap gap-2">
                                {message.sources.map((source, index) => (
                                  <li
                                    key={`${source.filename}-${index}`}
                                    className="inline-flex max-w-full items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1.5 text-xs text-slate-600"
                                    title={source.filename}
                                  >
                                    <svg viewBox="0 0 16 16" fill="none" className="size-3.5 shrink-0 text-emerald-700" aria-hidden="true">
                                      <path d="M4 1.75h5l3 3v9.5H4v-12.5Z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
                                      <path d="M9 1.9v3h2.8M6 8h4M6 10.5h4" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
                                    </svg>
                                    <span className="truncate font-medium">{source.title}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                          <div className="mt-3">
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              className="h-8 gap-2 rounded-lg border-0 bg-transparent px-2 text-xs font-medium text-emerald-800 hover:bg-emerald-50"
                              onClick={() => handleAnswerAudioClick(message)}
                              disabled={ttsLoadingId !== null}
                              aria-label={ttsLoadingId === message.id ? "Generating audio" : ttsPlayingId === message.id ? "Pause answer" : ttsPausedId === message.id ? "Resume answer" : "Play answer aloud"}
                            >
                              {ttsLoadingId === message.id ? (
                                <Volume2 className="size-4 animate-pulse" aria-hidden="true" />
                              ) : ttsPlayingId === message.id ? (
                                <Pause className="size-4" aria-hidden="true" />
                              ) : ttsPausedId === message.id ? (
                                <Play className="size-4" aria-hidden="true" />
                              ) : (
                                <Volume2 className="size-4" aria-hidden="true" />
                              )}
                              {ttsLoadingId === message.id ? "Preparing answer…" : ttsPlayingId === message.id ? "Pause" : ttsPausedId === message.id ? "Resume" : "Play answer"}
                            </Button>
                            {ttsError?.id === message.id && (
                              <p className="mt-1 text-xs text-rose-700" role="alert">{ttsError.message}</p>
                            )}
                          </div>
                        </>
                      )}
                    </div>
                  </article>
                ))}

                {isLoading && (
                  <div className="flex items-start gap-3 sm:gap-4" aria-label="Assistant is responding">
                    <div className="mt-1 flex size-9 shrink-0 items-center justify-center rounded-xl bg-emerald-800 text-white" aria-hidden="true">
                      <svg viewBox="0 0 24 24" fill="none" className="size-5">
                        <path d="M13.4 2.8 5.8 13h5l-.3 8.2L18.2 11h-5.1l.3-8.2Z" fill="currentColor" />
                      </svg>
                    </div>
                    <div className="flex items-center gap-3 pt-2 text-sm text-slate-500">
                      <span className="flex gap-1" aria-hidden="true">
                        <i className="size-1.5 animate-bounce rounded-full bg-emerald-700 [animation-delay:-0.2s]" />
                        <i className="size-1.5 animate-bounce rounded-full bg-emerald-700 [animation-delay:-0.1s]" />
                        <i className="size-1.5 animate-bounce rounded-full bg-emerald-700" />
                      </span>
                      Checking approved guidance…
                    </div>
                  </div>
                )}

                {error && (
                  <div className="ml-12 rounded-xl border border-rose-100 bg-rose-50 px-4 py-3 text-sm text-rose-800" role="alert">
                    {error}
                  </div>
                )}
                <div ref={bottomRef} />
              </div>
            )}
          </div>

          <div className="border-t border-slate-100 bg-white px-4 py-4 sm:px-7 sm:py-5">
            {messages.length === 0 && error && (
              <p className="mb-3 rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-800" role="alert">
                {error}
              </p>
            )}
            <form onSubmit={handleSubmit} className="mx-auto max-w-3xl">
              <label htmlFor="question" className="sr-only">Ask a question</label>
              <div className="flex min-w-0 items-center gap-3 rounded-2xl border border-slate-200 bg-slate-50 p-2 transition focus-within:border-emerald-400 focus-within:bg-white focus-within:ring-4 focus-within:ring-emerald-50">
                <textarea
                  ref={inputRef}
                  id="question"
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  onKeyDown={handleInputKeyDown}
                  placeholder="Ask a question about utility services…"
                  rows={1}
                  className="max-h-36 min-h-11 min-w-0 flex-1 resize-y bg-transparent px-3 py-3 text-sm leading-5 text-slate-800 outline-none placeholder:text-slate-400"
                />
                <Button
                  type="button"
                  variant="outline"
                  onClick={handleMicrophoneClick}
                  disabled={isTranscribing}
                  aria-label={isListening ? "Stop recording" : isTranscribing ? "Transcribing recording" : "Start recording"}
                  aria-pressed={isListening}
                  title={isListening ? "Stop recording" : isTranscribing ? "Transcribing" : "Start recording"}
                  className={`h-10 w-10 shrink-0 rounded-full border-slate-200 p-0 text-emerald-800 shadow-sm hover:bg-emerald-50 ${isListening ? "animate-pulse border-rose-500 bg-rose-600 text-white hover:bg-rose-700" : "bg-white"}`}
                >
                  <Mic className="size-4" aria-hidden="true" />
                </Button>
                <Button
                  type="submit"
                  disabled={isLoading || !question.trim()}
                  aria-label={isLoading ? "Sending message" : "Send message"}
                  title={isLoading ? "Sending" : "Send"}
                  className="h-10 w-10 shrink-0 rounded-full bg-emerald-800 p-0 text-white shadow-sm hover:bg-emerald-900 focus-visible:ring-emerald-600"
                >
                  <svg viewBox="0 0 20 20" fill="none" className="size-4" aria-hidden="true">
                    <path d="M3.2 10 16.8 3.5l-3.2 13-3.1-5.2L3.2 10Z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
                    <path d="m10.5 11.3 6.3-7.8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                  </svg>
                </Button>
              </div>
              <div className="mt-2 flex items-center justify-between px-1 text-[11px] text-slate-400">
                <span>{isListening ? "Listening… Click the microphone to stop." : isTranscribing && isLoading ? "Sending your voice question…" : isTranscribing ? "Transcribing recording…" : "Answers use approved organizational knowledge."}</span>
                <span className="hidden sm:inline">Enter to send · Shift + Enter for a new line</span>
              </div>
            </form>
          </div>
        </section>
      </div>
    </main>
  );
}
