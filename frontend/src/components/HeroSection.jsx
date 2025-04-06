import Header from "./Header";
import "./landing.css";

const HeroSection = () => {
  return (
    <div className="relative">
      <div
        className="absolute inset-0 blur-xl h-[580px] -z-10"
        style={{
          background:
            "linear-gradient(143.6deg, rgba(192, 132, 252, 0) 20.79%, rgba(232, 121, 249, 0.26) 40.92%, rgba(204, 171, 238, 0) 70.35%)",
        }}
      ></div>

      <Header />

      <section id="hero">
        <div className="max-w-screen-xl mx-auto px-4 sm:px-6 md:px-8 py-16 sm:py-24 flex flex-col-reverse md:flex-row items-center gap-12 text-center md:text-left">
          <div className="w-full max-w-xl space-y-6">
            <a
              href="/login"
              className="inline-flex gap-x-2 items-center text-sm font-medium border px-3 py-1.5 rounded-full hover:bg-white transition"
            >
              AI Powered Steganography Detection
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path d="M7.21 14.77a.75.75 0 01.02-1.06L11.17 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z" />
              </svg>
            </a>
            <h1 className="text-4xl sm:text-5xl font-extrabold text-gray-800">
              Protecting Your Data, One Pixel at a Time
            </h1>
            <p className="text-gray-600">
              StegoShield detects hidden payloads inside everyday media files — images, audio, and video — using cutting-edge AI.
            </p>
            <div className="flex flex-col sm:flex-row justify-center md:justify-start gap-3 sm:gap-4">
              <a
                href="/signup"
                className="inline-flex items-center justify-center gap-x-2 px-5 py-2 text-white bg-gray-800 hover:bg-gray-700 rounded-full text-sm"
              >
                Get started
              </a>
              <a
                href="#features"
                className="inline-flex items-center justify-center gap-x-2 px-5 py-2 text-gray-700 hover:text-gray-900 rounded-full text-sm"
              >
                Learn more
              </a>
            </div>
          </div>

          <div className="w-full md:w-1/2 flex justify-center">
            <img
              src="https://raw.githubusercontent.com/sidiDev/remote-assets/c86a7ae02ac188442548f510b5393c04140515d7/undraw_progressive_app_m-9-ms_oftfv5.svg"
              alt="Hero illustration"
              className="w-full max-w-md sm:max-w-xl"
            />
          </div>
        </div>
      </section>
    </div>
  );
};

export default HeroSection;
