
import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider,signInWithPopup } from "firebase/auth";
import { getAnalytics } from "firebase/analytics";

const firebaseConfig = {
    apiKey: "AIzaSyAQqvDYNSMWD5Pkm7e8Pf7cN5ujhZy6NqE",
    authDomain: "stego-shield.firebaseapp.com",
    projectId: "stego-shield",
    storageBucket: "stego-shield.appspot.com",
    messagingSenderId: "951143754468",
    appId: "1:951143754468:web:bf42b3e8b3b44b8b1ddaf1",
    measurementId: "G-90T3VFKP80"
  };

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const analytics = getAnalytics(app);
const provider = new GoogleAuthProvider();

export { auth, provider, signInWithPopup  };
