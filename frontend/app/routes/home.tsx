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
    <div className="min-h-screen flex flex-col items-center justify-between px-4 py-10">

      <div className="flex flex-col items-center my-10">
        <h1 className="text-2xl font-bold">Bem-Vindo!</h1>
        <p>Esse é o Loop, um categorizador de fotos com duas possibilidades (Cachorro e Gato).</p>
      </div>

      <div className="fixed z-50 top-0 left-0 right-200 p-2 bg-linear-to-r from-purple-500 via-gray-950 to-gray-950/50">
        <div className="flex justify-right items-center space-x-3">
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
              src="../../public/images/loop_dark.png" 
              alt="Kauã Repo" />
          </a>
          <Modal />
        </div>
      </div>

      <div className="flex items-center justify-between">
        <Predict_Card/>
      </div>

      <div className="flex">
        <p>Tecnologia <a target="_blank" className="font-bold underline decoration-green-300 cursor-pointer" href="https://github.com/Kaua-Matheus">Kaua-Matheus</a> </p>
      </div>
    </div>
  )
}
