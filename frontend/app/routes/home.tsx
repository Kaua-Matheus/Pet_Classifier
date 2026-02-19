import type { Route } from "./+types/home";

// Components
import Predict_Card from "~/assets/predict";
import Modal from "~/assets/modal";


export function meta({}: Route.MetaArgs) {

  return [
    { title: "Pet Classifier" },
    { 
      name: "description", 
      content: "Web application with the task to provide if your pet is a cat or a dog!" 
    },
  ];
}

export default function Home() {

  return (
    <div className="min-h-screen flex flex-col items-center justify-between px-2 sm:px-4 py-4 sm:py-10">

      <div className="flex flex-col items-center my-4 sm:my-10 text-center mt-12">
        <h1 className="text-xl sm:text-2xl font-bold text-white mb-2">Bem-Vindo!</h1>
        <p className="text-sm sm:text-base text-gray-300 px-4 leading-relaxed">
          <span className="hidden sm:inline">
            Esse é o Loop, um categorizador de fotos com duas possibilidades (Cachorro e Gato).
          </span>
          <span className="sm:hidden">
            Loop: Identifique se seu pet é um cachorro ou gato!
          </span>
        </p>
      </div>

      <div className="fixed z-50 top-0 left-0 right-0 p-2 sm:p-2 bg-gray-500 md:bg-linear-to-r from-purple-500 via-gray-950 to-gray-950/50">
        <div className="flex justify-between items-center space-x-2 sm:space-x-3 max-w-7xl mx-auto">
          <a 
            href="https://github.com/Kaua-Matheus/Pet_Classifier"
            target="_blank">
            <img 
              className="
                relative h-12
                rounded-lg transition-all duration-300 ease-in-out
                transform active:scale-95
                hover:-translate-y-0.5 hover:scale-102
                cursor-pointer shadow-lg hover:shadow-xl" 
              src="/images/loop_dark.png" 
              alt="Kauã Repo" />
          </a>
          <Modal />
        </div>
      </div>

      <div className="flex-1 flex items-center justify-center w-full pt-16 sm:pt-0">
        <Predict_Card/>
      </div>

      <div className="flex justify-center mt-6 sm:mt-0">
        <p className="text-xs sm:text-sm text-gray-400 text-center">
          <span className="hidden sm:inline">Tecnologia </span>
          <a 
            target="_blank" 
            className="font-bold underline decoration-green-300 cursor-pointer hover:text-green-300 transition-colors" 
            href="https://github.com/Kaua-Matheus"
          >
            Kaua-Matheus
          </a>
        </p>
      </div>
    </div>
  )
}
