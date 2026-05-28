'use client';
import { useState, useRef } from 'react';
const API = 'https://use-ai-malayalamai-production-ee70.up.railway.app';
const NGROK_HEADER = { 'ngrok-skip-browser-warning': 'true' };

const STYLES = [
  { key: 'standard',  label: 'Standard' },
  { key: 'formal',    label: 'Formal' },
  { key: 'casual',    label: 'Casual' },
  { key: 'news',      label: 'News' },
  { key: 'literary',  label: 'Literary' },
  { key: 'business',  label: 'Business' },
  { key: 'academic',  label: 'Academic' },
  { key: 'simple',    label: 'Simple' },
  { key: 'humorous',  label: 'Humorous' },
  { key: 'emotional', label: 'Emotional' },
  { key: 'bullet',    label: 'Bullet Points' },
];

export default function Home() {
  const [screen, setScreen] = useState('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [token, setToken] = useState(null);
  const [userEmail, setUserEmail] = useState('');
  const [authError, setAuthError] = useState('');
  const [recording, setRecording] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showPlans, setShowPlans] = useState(false);
  const [plans, setPlans] = useState([]);
  const [streamStatus, setStreamStatus] = useState('');
  const [englishLive, setEnglishLive] = useState('');
  const [malayalamLive, setMalayalamLive] = useState('');
  const [refinedText, setRefinedText] = useState('');
  const [isDone, setIsDone] = useState(false);
  const [selectedStyle, setSelectedStyle] = useState('standard');
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const resetResults = () => {
    setEnglishLive(''); setMalayalamLive(''); setRefinedText('');
    setStreamStatus(''); setIsDone(false); setError(null);
  };
  const handleRegister = async () => {
    setAuthError('');
    try {
      const res = await fetch(`${API}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...NGROK_HEADER },
        body: JSON.stringify({ email, password })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Registration failed');
      setToken(data.token); setUserEmail(data.email); setScreen('app');
    } catch (e) { setAuthError(e.message); }
  };
  const handleLogin = async () => {
    setAuthError('');
    try {
      const form = new URLSearchParams();
      form.append('username', email); form.append('password', password);
      const res = await fetch(`${API}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded', ...NGROK_HEADER },
        body: form
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Login failed');
      setToken(data.access_token); setUserEmail(data.email); setScreen('app');
    } catch (e) { setAuthError(e.message); }
  };
  const startRecording = async () => {
    resetResults();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];
      mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      mediaRecorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        await sendAudioStream(blob);
        stream.getTracks().forEach(t => t.stop());
      };
      mediaRecorder.start(250);
      setRecording(true);
    } catch (err) { setError('Microphone access denied.'); }
  };
  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setRecording(false); setLoading(true);
      setStreamStatus('Uploading audio...');
    }
  };
  const sendAudioStream = async (blob) => {
    try {
      const formData = new FormData();
      formData.append('file', blob, 'recording.webm');
      formData.append('style', selectedStyle);
      const res = await fetch(`${API}/audio/process-stream`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}`, ...NGROK_HEADER },
        body: formData,
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errData.detail || 'Server error');
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;
          let evt;
          try { evt = JSON.parse(raw); } catch { continue; }
          if (evt.type === 'status') { setStreamStatus(evt.message); setLoading(false); }
          else if (evt.type === 'english_segment') { setEnglishLive(evt.accumulated); setStreamStatus('Transcribing English...'); setLoading(false); }
          else if (evt.type === 'malayalam_segment') { setMalayalamLive(evt.accumulated); setStreamStatus('Transcribing Malayalam...'); }
          else if (evt.type === 'complete') { setEnglishLive(evt.english_text); setMalayalamLive(evt.malayalam_text); setRefinedText(evt.refined_text || evt.english_text); setStreamStatus(''); setIsDone(true); }
          else if (evt.type === 'error') { setError('Error: ' + evt.message); }
        }
      }
    } catch (err) { setError('Failed: ' + err.message); }
    finally { setLoading(false); }
  };
  const loadPlans = async () => {
    try {
      const res = await fetch(`${API}/payment/plans`, { headers: NGROK_HEADER });
      const data = await res.json();
      setPlans(data.plans); setShowPlans(true);
    } catch (e) { console.error('Failed to load plans'); }
  };
  const handlePayment = async (planId) => {
    try {
      const res = await fetch(`${API}/payment/create-order`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify({ plan: planId })
      });
      const order = await res.json();
      const options = {
        key: order.key_id, amount: order.amount, currency: order.currency,
        name: 'Malayalam Voice AI', description: `${planId} Plan`, order_id: order.order_id,
        handler: async (response) => {
          const verifyRes = await fetch(`${API}/payment/verify`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
            body: JSON.stringify({ razorpay_order_id: response.razorpay_order_id, razorpay_payment_id: response.razorpay_payment_id, razorpay_signature: response.razorpay_signature, plan: planId })
          });
          const result = await verifyRes.json();
          alert(result.message); setShowPlans(false);
        },
        theme: { color: '#16a34a' }
      };
      const rzp = new window.Razorpay(options); rzp.open();
    } catch (e) { alert('Payment failed. Please try again.'); }
  };
  if (screen === 'login' || screen === 'register') {
    return (
      <main className="min-h-screen bg-gray-950 text-white flex flex-col items-center justify-center p-6">
        <h1 className="text-3xl font-bold mb-2 text-green-400">Malayalam Voice AI</h1>
        <p className="text-gray-400 mb-8 text-sm">Speak Malayalam - get instant English translation</p>
        <div className="bg-gray-800 rounded-2xl p-8 w-full max-w-sm">
          <h2 className="text-xl font-semibold mb-6 text-center">{screen === 'login' ? 'Login' : 'Create Account'}</h2>
          <input type="email" placeholder="Email" value={email} onChange={e => setEmail(e.target.value)}
            className="w-full bg-gray-700 rounded-lg px-4 py-3 mb-3 text-white placeholder-gray-400 outline-none" />
          <input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)}
            className="w-full bg-gray-700 rounded-lg px-4 py-3 mb-4 text-white placeholder-gray-400 outline-none" />
          {authError && <p className="text-red-400 text-sm mb-4">{authError}</p>}
          <button onClick={screen === 'login' ? handleLogin : handleRegister}
            className="w-full bg-green-600 hover:bg-green-700 rounded-lg py-3 font-semibold mb-4">
            {screen === 'login' ? 'Login' : 'Register'}
          </button>
          <p className="text-center text-sm text-gray-400">
            {screen === 'login' ? "Don't have an account? " : "Already have an account? "}
            <span onClick={() => { setScreen(screen === 'login' ? 'register' : 'login'); setAuthError(''); }}
              className="text-green-400 cursor-pointer hover:underline">
              {screen === 'login' ? 'Register' : 'Login'}
            </span>
          </p>
        </div>
      </main>
    );
  }
  const hasContent = englishLive || malayalamLive || refinedText || streamStatus || loading;
  return (
    <main className="min-h-screen bg-gray-950 text-white flex flex-col items-center justify-center p-6">
      <div className="w-full max-w-md">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-2xl font-bold text-green-400">Malayalam Voice AI</h1>
          <div className="flex items-center gap-3">
            <span className="text-xs text-gray-400">{userEmail}</span>
            <div className="flex gap-2">
              <button onClick={loadPlans} className="text-xs text-yellow-400 hover:text-yellow-300 border border-yellow-600 rounded px-2 py-1">Upgrade</button>
              <button onClick={() => { setToken(null); setScreen('login'); resetResults(); }} className="text-xs text-gray-400 hover:text-white border border-gray-600 rounded px-2 py-1">Logout</button>
            </div>
          </div>
        </div>
        <div className="flex flex-col items-center mb-6">
          <button onClick={recording ? stopRecording : startRecording} disabled={loading}
            className={`w-28 h-28 rounded-full text-4xl transition-all duration-200 shadow-lg ${recording ? 'bg-red-600 hover:bg-red-700 animate-pulse' : loading ? 'bg-gray-600 cursor-not-allowed' : 'bg-green-600 hover:bg-green-700'}`}>
            {loading ? '⏳' : recording ? '⏹' : '🎤'}
          </button>
          <p className="mt-4 text-sm text-gray-400">
            {recording ? 'Recording... tap to stop' : loading ? 'Processing...' : 'Tap to start recording'}
          </p>
        </div>
        <div className="mb-6">
          <p className="text-xs text-gray-500 uppercase tracking-wide mb-2 text-center">Translation Style</p>
          <div className="flex flex-wrap gap-2 justify-center">
            {STYLES.map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setSelectedStyle(key)}
                disabled={recording || loading}
                className={`px-3 py-1 rounded-full text-xs font-medium border transition-all duration-150 ${
                  selectedStyle === key
                    ? 'bg-green-600 border-green-500 text-white'
                    : 'bg-gray-800 border-gray-600 text-gray-400 hover:border-gray-400 hover:text-gray-200'
                } ${recording || loading ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer'}`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
        {streamStatus && !isDone && (
          <div className="flex items-center gap-2 mb-4 text-sm text-yellow-300">
            <svg className="w-4 h-4 animate-spin shrink-0" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
            </svg>
            {streamStatus}
          </div>
        )}
        {error && <div className="bg-red-900 border border-red-600 rounded-lg p-4 mb-4 text-sm text-red-200">{error}</div>}
        {hasContent && (
          <div className="space-y-4">
            <div className="bg-gray-800 rounded-lg p-4">
              <p className="text-xs text-gray-400 mb-2 uppercase tracking-wide">English Transcript</p>
              <p className="text-white leading-relaxed min-h-6">
                {englishLive ? englishLive : <span className="text-gray-500 italic text-sm">Waiting for speech...</span>}
              </p>
            </div>
            <div className="bg-gray-800 rounded-lg p-4">
              <p className="text-xs text-green-400 mb-2 uppercase tracking-wide">
                English Translation · <span className="capitalize text-green-300">{selectedStyle}</span>
              </p>
              <p className="text-white leading-relaxed min-h-6">
                {refinedText ? refinedText : isDone ? englishLive : <span className="text-gray-500 italic text-sm">Available after transcription...</span>}
              </p>
            </div>
            <div className="bg-gray-800 rounded-lg p-4">
              <p className="text-xs text-blue-400 mb-2 uppercase tracking-wide">Malayalam (Unicode)</p>
              <p className="text-white text-xl leading-relaxed min-h-6">
                {malayalamLive ? malayalamLive : <span className="text-gray-500 italic text-sm">Available after English pass...</span>}
              </p>
            </div>
            {isDone && (
              <button onClick={resetResults} className="w-full text-sm text-gray-400 hover:text-white border border-gray-700 rounded-lg py-2 transition">
                Clear results
              </button>
            )}
          </div>
        )}
      </div>
      {showPlans && (
        <div className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center p-4 z-50">
          <div className="bg-gray-800 rounded-2xl p-6 w-full max-w-md">
            <h2 className="text-xl font-bold text-white mb-6 text-center">Choose Your Plan</h2>
            <div className="space-y-4">
              {plans.map(plan => (
                <div key={plan.id} className="bg-gray-700 rounded-xl p-4 flex justify-between items-center">
                  <div>
                    <p className="text-white font-semibold">{plan.name}</p>
                    <p className="text-green-400 text-lg font-bold">Rs.{plan.price}/month</p>
                    <p className="text-gray-400 text-xs mt-1">{plan.features.join(' - ')}</p>
                  </div>
                  <button onClick={() => handlePayment(plan.id)} className="bg-green-600 hover:bg-green-700 text-white rounded-lg px-4 py-2 text-sm font-semibold">Buy</button>
                </div>
              ))}
            </div>
            <button onClick={() => setShowPlans(false)} className="w-full mt-4 text-gray-400 hover:text-white text-sm">Cancel</button>
          </div>
        </div>
      )}
    </main>
  );
}
