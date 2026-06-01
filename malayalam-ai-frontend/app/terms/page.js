export default function TermsPage() {
  const sections = [
    { title: "1. Acceptance of Terms", body: "By registering and using Malayalam Voice AI (malayalam-ai-frontend.vercel.app), you agree to these Terms and Conditions. If you do not agree, please do not use the service." },
    { title: "2. Service Description", body: "Malayalam Voice AI is a speech-to-text and translation service that converts spoken audio in Malayalam, Tamil, Telugu, Kannada, and Hindi into English text and native Unicode script. The service is provided on a free and paid subscription basis." },
    { title: "3. User Accounts", body: "You must register with a valid email address. You are responsible for maintaining the security of your account credentials. We reserve the right to suspend accounts that violate these terms." },
    { title: "4. Free and Paid Plans", body: "The Free plan allows 10 recordings per day at no charge. Paid plans (Basic Rs.99/month, Pro Rs.249/month, Premium Rs.499/month) provide higher daily limits. No refunds are provided for partial months once a payment is processed." },
    { title: "5. Payments", body: "All payments are processed securely by Razorpay. By purchasing a plan, you authorise us to charge your selected payment method. Plan prices are in Indian Rupees (INR)." },
    { title: "6. Acceptable Use", body: "You agree not to use the service to record or transmit any content that is illegal, defamatory, harmful, or violates the rights of others. You must not attempt to reverse-engineer, overload, or misuse the service." },
    { title: "7. Privacy and Data", body: "Audio recordings are processed in real-time and are NOT stored on our servers after transcription. We store only your email address, hashed password, subscription plan, and usage count. We do not sell your data to third parties." },
    { title: "8. Third-Party Services", body: "The service uses Sarvam AI and Groq for speech recognition, Razorpay for payments, Railway for backend hosting, and Vercel for frontend hosting. Each provider's own terms apply to their respective services." },
    { title: "9. Accuracy Disclaimer", body: "Speech recognition accuracy depends on audio quality, microphone type, accent, and background noise. We do not guarantee 100% accuracy. The service is provided as-is without warranty of any kind." },
    { title: "10. Service Availability", body: "We aim for high availability but do not guarantee uninterrupted access. Maintenance or third-party outages may cause temporary downtime. We are not liable for losses arising from service unavailability." },
    { title: "11. Intellectual Property", body: "All content, code, and design of Malayalam Voice AI is owned by Muhammed Asarudheen M. You may not copy, reproduce, or distribute any part of the service without explicit written permission." },
    { title: "12. Limitation of Liability", body: "To the maximum extent permitted by law, Malayalam Voice AI and its owner shall not be liable for any indirect, incidental, or consequential damages arising from the use or inability to use the service." },
    { title: "13. Governing Law", body: "These terms are governed by the laws of India. Any disputes shall be subject to the exclusive jurisdiction of courts in Kerala, India." },
    { title: "14. Changes to Terms", body: "We reserve the right to update these terms at any time. Continued use of the service after changes constitutes acceptance of the updated terms." },
    { title: "15. Contact", body: "For questions about these terms, contact: mamcbt@gmail.com" },
  ];
  return (
    <main style={{ minHeight:"100vh", backgroundColor:"#0f0f0f", color:"#e2e8f0", padding:"40px 20px", fontFamily:"sans-serif" }}>
      <div style={{ maxWidth:720, margin:"0 auto" }}>
        <h1 style={{ fontSize:"1.8rem", fontWeight:800, color:"#10b981", marginBottom:8 }}>Terms and Conditions</h1>
        <p style={{ color:"#64748b", fontSize:"0.85rem", marginBottom:32 }}>Malayalam Voice AI &nbsp;|&nbsp; Effective: June 1, 2026 &nbsp;|&nbsp; Owner: Muhammed Asarudheen M</p>
        {sections.map(({ title, body }) => (
          <div key={title} style={{ marginBottom:20, padding:"16px 20px", background:"#1a1a1a", borderRadius:10, border:"1px solid #2a2a2a" }}>
            <h3 style={{ fontSize:"1rem", fontWeight:700, color:"#f1f5f9", marginBottom:8 }}>{title}</h3>
            <p style={{ fontSize:"0.88rem", color:"#94a3b8", lineHeight:1.7 }}>{body}</p>
          </div>
        ))}
        <div style={{ marginTop:32, padding:"16px 20px", background:"#0d2b1a", border:"1px solid #10b98140", borderRadius:10, textAlign:"center" }}>
          <p style={{ fontSize:"0.85rem", color:"#64748b" }}>By using Malayalam Voice AI, you confirm that you have read and agree to these Terms and Conditions.</p>
          <a href="/" style={{ display:"inline-block", marginTop:12, padding:"10px 24px", background:"#10b981", color:"#000", borderRadius:8, fontWeight:700, fontSize:"0.9rem", textDecoration:"none" }}>Back to App</a>
        </div>
      </div>
    </main>
  );
}
