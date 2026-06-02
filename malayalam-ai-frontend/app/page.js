'use client';
import { useState, useRef, useEffect } from 'react';

// ── Config ────────────────────────────────────────────────────────────────────
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL
  || 'https://use-ai-malayalamai-production-ee70.up.railway.app';
const API = (path) => `${API_BASE}${path}`;

// ── Constants ─────────────────────────────────────────────────────────────────
const LANGUAGE_LABELS = {
  ml: 'MALAYALAM UNICODE',
  ta: 'TAMIL UNICODE',
  te: 'TELUGU UNICODE',
  kn: 'KANNADA UNICODE',
  hi: 'HINDI UNICODE',
};

const LANGUAGES = [
  { key: 'ml', label: 'Malayalam' },
  { key: 'ta', label: 'Tamil' },
  { key: 'te', label: 'Telugu' },
  { key: 'kn', label: 'Kannada' },
  { key: 'hi', label: 'Hindi' },
];

const STYLES = [
  { key: 'standard',  label: 'Standard'  },
  { key: 'formal',    label: 'Formal'    },
  { key: 'casual',    label: 'Casual'    },
  { key: 'news',      label: 'News'      },
  { key: 'literary',  label: 'Literary'  },
  { key: 'business',  label: 'Business'  },
  { key: 'academic',  label: 'Academic'  },
  { key: 'simple',    label: 'Simple'    },
  { key: 'humorous',  label: 'Humorous'  },
  { key: 'emotional', label: 'Emotional' },
  { key: 'bullet',    label: 'Bullet Points' },
];

// ── Main Component ────────────────────────────────────────────────────────────
export default function Home() {
  // Auth state
  const [screen,      setScreen]      = useState('login');
  const [authMode,    setAuthMode]    = useState('login');
  const [email,       setEmail]       = useState('');
  const [password,    setPassword]    = useState('');
  const [token,       setToken]       = useState('');
  const [userEmail,   setUserEmail]   = useState('');
  const [authError,   setAuthError]   = useState('');
  const [forgotSent,  setForgotSent]  = useState(false);
  const [forgotLoad,  setForgotLoad]  = useState(false);

  // Recording state
  const [recording,     setRecording]     = useState(false);
  const [loading,       setLoading]       = useState(false);
  const [error,         setError]         = useState('');
  const [streamStatus,  setStreamStatus]  = useState('');
  const [isDone,        setIsDone]        = useState(false);

  // Results
  const [englishLive,  setEnglishLive]  = useState('');
  const [nativeLive,   setNativeLive]   = useState('');
  const [refinedText,  setRefinedText]  = useState('');
  const [nativeLabel,  setNativeLabel]  = useState('MALAYALAM UNICODE');
  const [copiedBox,    setCopiedBox]    = useState(null);

  // Settings
  const [selectedLang,  setSelectedLang]  = useState('ml');
  const [selectedStyle, setSelectedStyle] = useState('standard');

  // Plans
  const [showPlans, setShowPlans] = useState(false);
  const [plans,     setPlans]     = useState([]);

  // Refs
  const mediaRecorderRef = useRef(null);
  const mediaStreamRef   = useRef(null);
  const chunksRef        = useRef([]);

  // ── Persist token across refresh ──────────────────────────────────────────
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const t = localStorage.getItem('diya_token');
    const e = localStorage.getItem('diya_email');
    if (t && e) { setToken(t); setUserEmail(e); setScreen('app'); }
  }, []);

  // ── Sync native label when language changes ───────────────────────────────
  useEffect(() => {
    setNativeLabel(LANGUAGE_LABELS[selectedLang] || 'NATIVE UNICODE');
  }, [selectedLang]);

  // ── Helpers ───────────────────────────────────────────────────────────────
  const resetResults = () => {
    setEnglishLive(''); setNativeLive(''); setRefinedText('');
    setStreamStatus(''); setIsDone(false); setError('');
    setNativeLabel(LANGUAGE_LABELS[selectedLang] || 'NATIVE UNICODE');
  };

  const copyText = async (text, box) => {
    try { await navigator.clipboard.writeText(text); } catch {}
    setCopiedBox(box);
    setTimeout(() => setCopiedBox(null), 2000);
  };

  // ── Auth ──────────────────────────────────────────────────────────────────
  const handleLogin = async (e) => {
    if (e) e.preventDefault();
    if (loading) return;
    setAuthError(''); setLoading(true);
    try {
      const form = new URLSearchParams();
      form.append('username', email.trim());
      form.append('password', password);
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 15000); // 15s timeout
      const res = await fetch(API('/auth/login'), {
        method:  'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body:    form,
        signal:  controller.signal,
      });
      clearTimeout(timeout);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Login failed');
      localStorage.setItem('diya_token', data.access_token);
      localStorage.setItem('diya_email', data.email);
      setToken(data.access_token); setUserEmail(data.email); setScreen('app');
    } catch (err) {
      if (err.name === 'AbortError') {
        setAuthError('Connection timed out. Check your internet and try again.');
      } else {
        setAuthError(err.message);
      }
    }
    finally { setLoading(false); }
  };

  const handleRegister = async (e) => {
    if (e) e.preventDefault();
    if (loading) return;
    setAuthError(''); setLoading(true);
    try {
      const res  = await fetch(API('/auth/register'), {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ email: email.trim(), password }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Registration failed');
      localStorage.setItem('diya_token', data.token || data.access_token);
      localStorage.setItem('diya_email', data.email);
      setToken(data.token || data.access_token); setUserEmail(data.email); setScreen('app');
    } catch (err) { setAuthError(err.message); }
    finally { setLoading(false); }
  };

  const handleForgotPassword = async () => {
    if (!email) { setAuthError('Enter your email address first.'); return; }
    setForgotLoad(true); setAuthError('');
    try {
      const res = await fetch(API('/auth/forgot-password'), {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ email: email.trim() }),
      });
      if (res.ok) setForgotSent(true);
      else setAuthError('Could not send reset email.');
    } catch { setAuthError('Could not reach server.'); }
    finally { setForgotLoad(false); }
  };

  const handleLogout = () => {
    localStorage.removeItem('diya_token');
    localStorage.removeItem('diya_email');
    setToken(''); setUserEmail(''); setEmail(''); setPassword('');
    setAuthError(''); setForgotSent(false); setShowPlans(false);
    setScreen('login'); resetResults();
  };

  // ── Safari-safe MIME detection ────────────────────────────────────────────
  const getSupportedMimeType = () => {
    const candidates = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/mp4;codecs=mp4a.40.2',
      'audio/mp4',
      'audio/aac',
      'audio/ogg;codecs=opus',
    ];
    for (const type of candidates) {
      if (window.MediaRecorder?.isTypeSupported?.(type)) return type;
    }
    return ''; // let browser choose
  };

  // ── Recording ─────────────────────────────────────────────────────────────
  const startRecording = async () => {
    resetResults();
    try {
      const stream   = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      const mimeType = getSupportedMimeType();
      const mr       = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);
      mediaRecorderRef.current = mr;
      chunksRef.current = [];
      mr.ondataavailable = (e) => { if (e.data && e.data.size > 0) chunksRef.current.push(e.data); };
      mr.onstop = async () => {
        const finalType = mimeType || chunksRef.current[0]?.type || 'audio/webm';
        const blob = new Blob(chunksRef.current, { type: finalType });
        setLoading(true); setStreamStatus('Uploading audio...');
        await sendAudio(blob);
        if (mediaStreamRef.current) {
          mediaStreamRef.current.getTracks().forEach(t => t.stop());
          mediaStreamRef.current = null;
        }
      };
      mr.start(250);
      setRecording(true);
    } catch {
      setError('Microphone access denied or recording is not supported on this device.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    setRecording(false);
  };

  // ── Audio submission (plain JSON — no SSE reader) ─────────────────────────
  const sendAudio = async (blob) => {
    try {
      setError(''); setIsDone(false);
      setStreamStatus('Processing your speech...');

      const formData = new FormData();
      formData.append('file',  blob, 'recording.webm');
      formData.append('style', selectedStyle);
      formData.append('lang',  selectedLang);

      const res  = await fetch(API('/audio/process'), {
        method:  'POST',
        headers: { Authorization: `Bearer ${token}` },
        body:    formData,
      });
      const data = await res.json();

      if (!res.ok) throw new Error(data.detail || data.error || 'Server error');

      const eng    = data.english_text || '';
      const native = data.native_text  || '';
      const label  = data.source_language_name
        ? `${data.source_language_name.toUpperCase()} UNICODE`
        : (LANGUAGE_LABELS[selectedLang] || 'NATIVE UNICODE');

      if (!eng && !native) {
        setError('No speech detected. Please speak clearly for 5–10 seconds and try again.');
        return;
      }

      setEnglishLive(eng);
      setNativeLive(native);
      setRefinedText(data.refined_text || eng);
      setNativeLabel(label);
      setStreamStatus('');
      setIsDone(true);

    } catch (err) {
      setError(`Failed: ${err.message}`);
    } finally {
      setLoading(false);
      setStreamStatus('');
    }
  };

  // ── Payment ───────────────────────────────────────────────────────────────
  const loadPlans = async () => {
    try {
      const res  = await fetch(API('/payment/plans'));
      const data = await res.json();
      setPlans(data.plans); setShowPlans(true);
    } catch { console.error('Failed to load plans'); }
  };

  const handlePayment = async (planId) => {
    if (!window.Razorpay) {
      alert('Payment system is still loading. Please try again in a moment.');
      return;
    }
    try {
      const order = await fetch(API('/payment/create-order'), {
        method:  'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body:    JSON.stringify({ plan: planId }),
      }).then(r => r.json());
      new window.Razorpay({
        key: order.key_id, amount: order.amount, currency: order.currency,
        name: 'Diya Voice AI', order_id: order.order_id,
        handler: async (resp) => {
          const v = await fetch(API('/payment/verify'), {
            method:  'POST',
            headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
            body:    JSON.stringify({ ...resp, plan: planId }),
          }).then(r => r.json());
          alert(v.message); setShowPlans(false);
        },
        theme: { color: '#16a34a' },
      }).open();
    } catch { alert('Payment failed. Please try again.'); }
  };

  // ── LOGIN / REGISTER SCREEN ───────────────────────────────────────────────
  if (screen !== 'app') {
    return (
      <main className="min-h-screen bg-gray-950 text-white flex flex-col items-center justify-center p-6">
        <h1 className="text-3xl font-bold mb-1 text-green-400">Diya Voice AI</h1>
        <p className="text-gray-400 mb-8 text-sm">Speak any Indian language — get instant English translation</p>

        <div className="bg-gray-800 rounded-2xl p-8 w-full max-w-sm">
          {/* Toggle */}
          <div className="flex rounded-xl bg-gray-700 p-1 mb-6">
            {['login', 'register'].map(m => (
              <button key={m} type="button"
                onClick={() => { setAuthMode(m); setAuthError(''); setForgotSent(false); }}
                className={`flex-1 py-2 text-sm font-medium rounded-lg transition capitalize
                  ${authMode === m ? 'bg-green-600 text-white' : 'text-gray-400 hover:text-white'}`}>
                {m}
              </button>
            ))}
          </div>

          <form onSubmit={authMode === 'login' ? handleLogin : handleRegister} autoComplete="on">
            <input
              type="email" name="email" placeholder="Email"
              value={email} onChange={e => setEmail(e.target.value)}
              autoComplete="email" inputMode="email" required
              className="w-full bg-gray-700 rounded-lg px-4 py-3 mb-3 text-white placeholder-gray-400 outline-none focus:ring-2 focus:ring-green-500" />
            <input
              type="password" name="password" placeholder="Password"
              value={password} onChange={e => setPassword(e.target.value)}
              autoComplete={authMode === 'login' ? 'current-password' : 'new-password'} required
              className="w-full bg-gray-700 rounded-lg px-4 py-3 mb-4 text-white placeholder-gray-400 outline-none focus:ring-2 focus:ring-green-500" />

            {authError && <p className="text-red-400 text-sm mb-4">{authError}</p>}
            {forgotSent && <p className="text-green-400 text-sm mb-4 text-center">Reset link sent — check your inbox.</p>}

            <button type="submit" disabled={loading}
              className="w-full bg-green-600 hover:bg-green-700 active:bg-green-800 disabled:opacity-50 rounded-lg py-3 font-semibold mb-3 transition">
              {loading ? 'Please wait...' : authMode === 'login' ? 'Login' : 'Register'}
            </button>
          </form>

          {authMode === 'login' && (
            <p className="text-center text-sm">
              <span onClick={handleForgotPassword}
                className={`text-gray-500 hover:text-gray-300 cursor-pointer underline text-xs
                  ${forgotLoad ? 'opacity-50 pointer-events-none' : ''}`}>
                {forgotLoad ? 'Sending...' : 'Forgot password?'}
              </span>
            </p>
          )}
        </div>
      </main>
    );
  }

  // ── MAIN APP ──────────────────────────────────────────────────────────────
  const hasContent = englishLive || nativeLive || refinedText || streamStatus || loading || error;

  return (
    <main className="min-h-screen bg-gray-950 text-white flex flex-col items-center p-6 pb-16">
      <div className="w-full max-w-md">

        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-2xl font-bold text-green-400">Diya Voice AI</h1>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500 hidden sm:block">{userEmail}</span>
            <button onClick={loadPlans}
              className="text-xs text-yellow-400 border border-yellow-600 rounded px-2 py-1 hover:bg-yellow-400/10 transition">
              ⭐ Upgrade
            </button>
            <button onClick={handleLogout}
              className="text-xs text-gray-400 border border-gray-600 rounded px-2 py-1 hover:bg-gray-700 transition">
              Logout
            </button>
          </div>
        </div>

        {/* Record button */}
        <div className="flex flex-col items-center mb-6">
          <button
            onClick={recording ? stopRecording : startRecording}
            disabled={loading}
            className={`w-28 h-28 rounded-full text-4xl transition-all duration-200 shadow-lg select-none
              ${recording ? 'bg-red-600 hover:bg-red-700 animate-pulse'
              : loading   ? 'bg-gray-600 cursor-not-allowed opacity-60'
              :             'bg-green-600 hover:bg-green-700 active:scale-95'}`}>
            {loading ? '⏳' : recording ? '⏹' : '🎤'}
          </button>
          <p className="mt-3 text-sm text-gray-400">
            {recording ? 'Recording… tap to stop' : loading ? 'Processing…' : 'Tap to start recording'}
          </p>
        </div>

        {/* Language selector */}
        <div className="mb-4">
          <p className="text-xs text-gray-500 uppercase tracking-wide mb-2 text-center">INPUT LANGUAGE</p>
          <div className="flex flex-wrap gap-2 justify-center">
            {LANGUAGES.map(l => (
              <button key={l.key}
                onClick={() => { setSelectedLang(l.key); resetResults(); }}
                disabled={recording || loading}
                className={`px-4 py-1.5 rounded-full text-sm border transition-all
                  ${selectedLang === l.key
                    ? 'bg-blue-500 text-white border-blue-500 font-semibold'
                    : 'bg-transparent text-gray-400 border-gray-600 hover:border-gray-400'}
                  ${(recording || loading) ? 'opacity-40 cursor-not-allowed' : ''}`}>
                {l.label}
              </button>
            ))}
          </div>
        </div>

        {/* Style selector */}
        <div className="mb-6">
          <p className="text-xs text-gray-500 uppercase tracking-wide mb-2 text-center">TRANSLATION STYLE</p>
          <div className="flex flex-wrap gap-2 justify-center">
            {STYLES.map(s => (
              <button key={s.key}
                onClick={() => { setSelectedStyle(s.key); resetResults(); }}
                disabled={recording || loading}
                className={`px-3 py-1 rounded-full text-xs font-medium border transition-all
                  ${selectedStyle === s.key
                    ? 'bg-green-600 border-green-500 text-white'
                    : 'bg-gray-800 border-gray-600 text-gray-400 hover:border-gray-400 hover:text-gray-200'}
                  ${(recording || loading) ? 'opacity-40 cursor-not-allowed' : ''}`}>
                {s.label}
              </button>
            ))}
          </div>
        </div>

        {/* Status pill */}
        {streamStatus && !isDone && (
          <div className="flex items-center gap-2 mb-4 text-sm text-yellow-300">
            <svg className="w-4 h-4 animate-spin shrink-0" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
            </svg>
            {streamStatus}
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="bg-red-900/50 border border-red-600 rounded-lg p-4 mb-4 text-sm text-red-200">
            ⚠️ {error}
          </div>
        )}

        {/* Output boxes */}
        {hasContent && (
          <div className="space-y-4">

            {/* English Transcript */}
            <div className="bg-gray-800 rounded-lg overflow-hidden">
              <div className="flex justify-between items-center px-4 py-2 border-b border-gray-700">
                <p className="text-xs text-gray-400 uppercase tracking-wide font-semibold">English Transcript</p>
                {englishLive && (
                  <button onClick={() => copyText(englishLive, 'en')}
                    className="text-xs text-gray-500 hover:text-white transition">
                    {copiedBox === 'en' ? '✓ Copied' : 'Copy'}
                  </button>
                )}
              </div>
              <p className="px-4 py-3 text-white leading-relaxed min-h-[48px] text-sm">
                {englishLive || <span className="text-gray-500 italic">Waiting for speech...</span>}
              </p>
            </div>

            {/* English Translation (Refined) */}
            <div className="bg-gray-800 rounded-lg overflow-hidden">
              <div className="flex justify-between items-center px-4 py-2 border-b border-gray-700">
                <p className="text-xs text-green-400 uppercase tracking-wide font-semibold">
                  English Translation · <span className="capitalize">{selectedStyle}</span>
                </p>
                {(refinedText || (isDone && englishLive)) && (
                  <button onClick={() => copyText(refinedText || englishLive, 'tr')}
                    className="text-xs text-gray-500 hover:text-white transition">
                    {copiedBox === 'tr' ? '✓ Copied' : 'Copy'}
                  </button>
                )}
              </div>
              <p className="px-4 py-3 text-white leading-relaxed min-h-[48px] text-sm">
                {refinedText || (isDone ? englishLive : null)
                  || <span className="text-gray-500 italic">Available after transcription...</span>}
              </p>
            </div>

            {/* Native Unicode */}
            <div className="bg-gray-800 rounded-lg overflow-hidden">
              <div className="flex justify-between items-center px-4 py-2 border-b border-gray-700">
                <p className="text-xs text-blue-400 uppercase tracking-wide font-semibold">
                  {nativeLabel}
                </p>
                {nativeLive && (
                  <button onClick={() => copyText(nativeLive, 'ml')}
                    className="text-xs text-gray-500 hover:text-white transition">
                    {copiedBox === 'ml' ? '✓ Copied' : 'Copy'}
                  </button>
                )}
              </div>
              <p className="px-4 py-3 text-white text-xl leading-loose min-h-[64px] ml-text">
                {nativeLive || <span className="text-gray-500 italic text-sm">Native script appears here...</span>}
              </p>
            </div>

            {/* Actions */}
            {isDone && (
              <button onClick={resetResults}
                className="w-full text-sm text-gray-400 hover:text-white border border-gray-700 rounded-lg py-2.5 transition hover:bg-gray-800">
                🗑️ Clear results
              </button>
            )}
          </div>
        )}
      </div>

      {/* Plans modal */}
      {showPlans && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-gray-800 rounded-2xl p-6 w-full max-w-md">
            <h2 className="text-xl font-bold text-white mb-6 text-center">Choose Your Plan</h2>
            <div className="space-y-3">
              {plans.map(plan => (
                <div key={plan.id} className="bg-gray-700 rounded-xl p-4 flex justify-between items-center">
                  <div>
                    <p className="text-white font-semibold">{plan.name}</p>
                    <p className="text-green-400 text-lg font-bold">₹{plan.price}/month</p>
                    <p className="text-gray-400 text-xs mt-1">{plan.features?.join(' · ')}</p>
                  </div>
                  <button onClick={() => handlePayment(plan.id)}
                    className="bg-green-600 hover:bg-green-700 text-white rounded-lg px-4 py-2 text-sm font-semibold transition">
                    Buy
                  </button>
                </div>
              ))}
            </div>
            <button onClick={() => setShowPlans(false)}
              className="w-full mt-4 text-gray-400 hover:text-white text-sm transition">
              Cancel
            </button>
          </div>
        </div>
      )}
    </main>
  );
}
