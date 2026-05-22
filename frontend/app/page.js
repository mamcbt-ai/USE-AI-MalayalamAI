'use client';
import { useState, useRef } from 'react';

const API = 'https://fester-yonder-stoplight.ngrok-free.dev';
const NGROK_HEADER = { 'ngrok-skip-browser-warning': 'true' };

export default function Home() {
  const [screen, setScreen] = useState('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [token, setToken] = useState(null);
  const [userEmail, setUserEmail] = useState('');
  const [authError, setAuthError] = useState('');
  const [recording, setRecording] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showPlans, setShowPlans] = useState(false);
  const [plans, setPlans] = useState([]);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

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
      setToken(data.token);
      setUserEmail(data.email);
      setScreen('app');
    } catch (e) {
      setAuthError(e.message);
    }
  };

  const handleLogin = async () => {
    setAuthError('');
    try {
      const form = new URLSearchParams();
      form.append('username', email);
      form.append('password', password);
      const res = await fetch(`${API}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded', ...NGROK_HEADER },
        body: form
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Login failed');
      setToken(data.access_token);
      setUserEmail(data.email);
      setScreen('app');
    } catch (e) {
      setAuthError(e.message);
    }
  };

  const startRecording = async () => {
    setError(null);
    setResult(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];
      mediaRecorder.ondataavailable = (e) => chunksRef.current.push(e.data);
      mediaRecorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        await sendAudio(blob);
        stream.getTracks().forEach(t => t.stop());
      };
      mediaRecorder.start();
      setRecording(true);
    } catch (err) {
      setError('Microphone access denied.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setRecording(false);
      setLoading(true);
    }
  };

  const sendAudio = async (blob) => {
    try {
      const formData = new FormData();
      formData.append('file', blob, 'recording.webm');
      const res = await fetch(`${API}/audio/process`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}`, ...NGROK_HEADER },
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError('Failed to process audio.');
    } finally {
      setLoading(false);
    }
  };

  const loadPlans = async () => {
    try {
      const res = await fetch(`${API}/payment/plans`, { headers: NGROK_HEADER });
      const data = await res.json();
      setPlans(data.plans);
      setShowPlans(true);
    } catch (e) {
      console.error('Failed to load plans');
    }
  };

  const handlePayment = async (planId) => {
    try {
      const res = await fetch(`${API}/payment/create-order`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ plan: planId })
      });
      const order = await res.json();
      const options = {
        key: order.key_id,
        amount: order.amount,
        currency: order.currency,
        name: 'Malayalam Voice AI',
        description: `${planId} Plan`,
        order_id: order.order_id,
        handler: async (response) => {
          const verifyRes = await fetch(`${API}/payment/verify`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
              razorpay_order_id: response.razorpay_order_id,
              razorpay_payment_id: response.razorpay_payment_id,
              razorpay_signature: response.razorpay_signature,
              plan: planId
            })
          });
          const result = await verifyRes.json();
          alert(result.message);
          setShowPlans(false);
        },
        theme: { color: '#16a34a' }
      };
      const rzp = new window.Razorpay(options);
      rzp.open();
    } catch (e) {
      alert('Payment failed. Please try again.');
    }
  };

  // LOGIN / REGISTER SCREEN
  if (screen === 'login' || screen === 'register') {
    return (
      <main className="min-h-screen bg-gray-950 text-white flex flex-col items-center justify-center p-6">
        <h1 className="text-3xl font-bold mb-2 text-green-400">Malayalam Voice AI</h1>
        <p className="text-gray-400 mb-8 text-sm">Speak Malayalam — get instant English translation</p>
        <div className="bg-gray-800 rounded-2xl p-8 w-full max-w-sm">
          <h2 className="text-xl font-semibold mb-6 text-center">
            {screen === 'login' ? 'Login' : 'Create Account'}
          </h2>
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={e => setEmail(e.target.value)}
            className="w-full bg-gray-700 rounded-lg px-4 py-3 mb-3 text-white placeholder-gray-400 outline-none"
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            className="w-full bg-gray-700 rounded-lg px-4 py-3 mb-4 text-white placeholder-gray-400 outline-none"
          />
          {authError && <p className="text-red-400 text-sm mb-4">{authError}</p>}
          <button
            onClick={screen === 'login' ? handleLogin : handleRegister}
            className="w-full bg-green-600 hover:bg-green-700 rounded-lg py-3 font-semibold mb-4"
          >
            {screen === 'login' ? 'Login' : 'Register'}
          </button>
          <p className="text-center text-sm text-gray-400">
            {screen === 'login' ? "Don't have an account? " : "Already have an account? "}
            <span
              onClick={() => { setScreen(screen === 'login' ? 'register' : 'login'); setAuthError(''); }}
              className="text-green-400 cursor-pointer hover:underline"
            >
              {screen === 'login' ? 'Register' : 'Login'}
            </span>
          </p>
        </div>
      </main>
    );
  }

  // MAIN APP SCREEN
  return (
    <main className="min-h-screen bg-gray-950 text-white flex flex-col items-center justify-center p-6">
      <div className="w-full max-w-md">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-2xl font-bold text-green-400">Malayalam Voice AI</h1>
          <div className="flex items-center gap-3">
            <span className="text-xs text-gray-400">{userEmail}</span>
            <div className="flex gap-2">
              <button
                onClick={loadPlans}
                className="text-xs text-yellow-400 hover:text-yellow-300 border border-yellow-600 rounded px-2 py-1"
              >
                Upgrade
              </button>
              <button
                onClick={() => { setToken(null); setScreen('login'); setResult(null); }}
                className="text-xs text-gray-400 hover:text-white border border-gray-600 rounded px-2 py-1"
              >
                Logout
              </button>
            </div>
          </div>
        </div>

        <div className="flex flex-col items-center mb-8">
          <button
            onClick={recording ? stopRecording : startRecording}
            disabled={loading}
            className={`w-28 h-28 rounded-full text-4xl transition-all duration-200 shadow-lg
              ${recording ? 'bg-red-600 hover:bg-red-700 animate-pulse'
                : loading ? 'bg-gray-600 cursor-not-allowed'
                : 'bg-green-600 hover:bg-green-700'}`}
          >
            {loading ? '⏳' : recording ? '⏹' : '🎤'}
          </button>
          <p className="mt-4 text-sm text-gray-400">
            {recording ? 'Recording... tap to stop' : loading ? 'Processing...' : 'Tap to start recording'}
          </p>
        </div>

        {error && (
          <div className="bg-red-900 border border-red-600 rounded-lg p-4 mb-4 text-sm text-red-200">{error}</div>
        )}

        {result && (
          <div className="space-y-4">
            <div className="bg-gray-800 rounded-lg p-4">
              <p className="text-xs text-gray-400 mb-1">Transcript</p>
              <p className="text-white">{result.asr_output?.text || result.asr_output?.segments?.[0]?.text || '-'}</p>
            </div>
            <div className="bg-gray-800 rounded-lg p-4">
              <p className="text-xs text-green-400 mb-1">English Translation</p>
              <p className="text-white">{result.translation_output?.translation || '-'}</p>
            </div>
            <div className="bg-gray-800 rounded-lg p-4">
              <p className="text-xs text-blue-400 mb-1">Malayalam (Unicode)</p>
              <p className="text-white">{result.reverse_translation?.malayalam || '-'}</p>
            </div>
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
                    <p className="text-green-400 text-lg font-bold">₹{plan.price}/month</p>
                    <p className="text-gray-400 text-xs mt-1">{plan.features.join(' • ')}</p>
                  </div>
                  <button
                    onClick={() => handlePayment(plan.id)}
                    className="bg-green-600 hover:bg-green-700 text-white rounded-lg px-4 py-2 text-sm font-semibold"
                  >
                    Buy
                  </button>
                </div>
              ))}
            </div>
            <button
              onClick={() => setShowPlans(false)}
              className="w-full mt-4 text-gray-400 hover:text-white text-sm"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </main>
  );
}