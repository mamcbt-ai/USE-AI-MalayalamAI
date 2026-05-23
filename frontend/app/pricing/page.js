"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

const API_URL = "https://use-ai-malayalamai-production-ee70.up.railway.app";

const plans = [
  { id: "basic", name: "Basic", price: 99, recordings: "50 recordings/day", color: "#4CAF50" },
  { id: "pro", name: "Pro", price: 249, recordings: "200 recordings/day", color: "#2196F3", badge: "Most Popular" },
  { id: "premium", name: "Premium", price: 499, recordings: "Unlimited recordings", color: "#9C27B0" },
];

export default function PricingPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(null);
  const [message, setMessage] = useState("");
  const [token, setToken] = useState(null);

  useEffect(() => {
    const stored = localStorage.getItem("token");
    if (!stored) { router.push("/login"); return; }
    setToken(stored);
    const script = document.createElement("script");
    script.src = "https://checkout.razorpay.com/v1/checkout.js";
    script.async = true;
    document.body.appendChild(script);
  }, []);

  const handleBuyNow = async (plan) => {
    setLoading(plan.id);
    setMessage("");
    try {
      const res = await fetch(API_URL + "/payment/create-order", {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: "Bearer " + token },
        body: JSON.stringify({ plan: plan.id }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to create order");
      const options = {
        key: data.key_id, amount: data.amount, currency: data.currency,
        name: "Malayalam Voice AI", order_id: data.order_id,
        handler: async function (response) {
          const verifyRes = await fetch(API_URL + "/payment/verify", {
            method: "POST",
            headers: { "Content-Type": "application/json", Authorization: "Bearer " + token },
            body: JSON.stringify({ razorpay_order_id: response.razorpay_order_id, razorpay_payment_id: response.razorpay_payment_id, razorpay_signature: response.razorpay_signature, plan: plan.id }),
          });
          const v = await verifyRes.json();
          setMessage(v.message ? "Payment successful! " + v.message : "Verification failed");
          if (v.message) setTimeout(() => router.push("/"), 2500);
        },
        theme: { color: plan.color },
        modal: { ondismiss: () => setLoading(null) },
      };
      new window.Razorpay(options).open();
    } catch (err) {
      setMessage("Error: " + err.message);
      setLoading(null);
    }
  };

  return (
    <div style={{ minHeight: "100vh", background: "#0f0f0f", color: "#fff", padding: "40px 20px", fontFamily: "sans-serif", textAlign: "center" }}>
      <h1 style={{ fontSize: "2rem", marginBottom: "8px" }}>Upgrade Your Plan</h1>
      <p style={{ color: "#aaa", marginBottom: "32px" }}>Free: 10 recordings/day. Upgrade for more.</p>
      {message && (<div style={{ background: "#1e1e1e", border: "1px solid #333", borderRadius: "8px", padding: "12px 20px", marginBottom: "24px" }}>{message}</div>)}
      <div style={{ display: "flex", gap: "24px", justifyContent: "center", flexWrap: "wrap", maxWidth: "900px", margin: "0 auto" }}>
        {plans.map((plan) => (
          <div key={plan.id} style={{ background: "#1a1a1a", border: "1px solid #333", borderRadius: "16px", padding: "32px 24px", width: "260px", position: "relative" }}>
            {plan.badge && (<div style={{ position: "absolute", top: "-12px", left: "50%", transform: "translateX(-50%)", background: "#2196F3", color: "#fff", padding: "4px 14px", borderRadius: "20px", fontSize: "12px", fontWeight: "bold" }}>{plan.badge}</div>)}
            <h2 style={{ fontSize: "1.4rem", marginBottom: "12px" }}>{plan.name}</h2>
            <div style={{ marginBottom: "8px" }}>
              <span style={{ fontSize: "1.1rem" }}>Rs. </span>
              <span style={{ fontSize: "3rem", fontWeight: "bold" }}>{plan.price}</span>
              <span style={{ color: "#aaa", fontSize: "0.9rem" }}>/month</span>
            </div>
            <p style={{ color: "#4CAF50", fontWeight: "bold", marginBottom: "20px" }}>{plan.recordings}</p>
            <button onClick={() => handleBuyNow(plan)} disabled={loading === plan.id}
              style={{ width: "100%", padding: "14px", border: "none", borderRadius: "8px", background: plan.color, color: "#fff", fontSize: "1rem", fontWeight: "bold", cursor: "pointer", opacity: loading === plan.id ? 0.7 : 1 }}>
              {loading === plan.id ? "Processing..." : "Buy Now"}
            </button>
          </div>
        ))}
      </div>
      <p style={{ color: "#555", marginTop: "32px", fontSize: "0.85rem" }}>Secure payment via Razorpay</p>
    </div>
  );
}