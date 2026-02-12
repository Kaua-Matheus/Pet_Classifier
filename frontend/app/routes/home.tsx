import type { Route } from "./+types/home";

// Components
import Predict_Card from "~/assets/predict";
import Button from "~/assets/button";

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

  const handleGetModel = async () => {
    console.log("Clicado!")
    try {
      const response = await fetch("http://192.168.18.21:8000/predict/", {
        method: "GET",
      })
      const data = await response.json()

      console.log(data)
    } catch (err) {
      console.error("Error trying to get the model info: ", err);
    }
  }

  const handleImageSelect = (file: File) => {
    console.log("Image selecionada: ", file.name);
  }

  const handlePredict = async (file: File) => {
    console.log(file.name)
    const formData = new FormData();
    formData.append("file", file);

    console.log(formData)

    try {
      const response = await fetch("http://192.168.18.21:8000/predict/", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();
      console.log("Predição: ", result);
    } catch (err) {
      console.error("Generic Error: ", err);
    }

  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-between px-4 py-10">
      <div className="flex flex-col items-center my-10">
        <h1 className="text-2xl font-bold">Bem-Vindo!</h1>
        <p>Esse é o Loop, um categorizador de fotos com duas possibilidades (Cachorro e Gato).</p>
        <Button onClick={handleGetModel}>Get Model</Button>
      </div>

      <div className="flex flex-col items-center">
        <Predict_Card onImageSelect={handleImageSelect} onPredict={handlePredict}/>
      </div>

      <div className="flex">
        <p>Tecnologia <a target="_blank" className="font-bold underline decoration-green-300 cursor-pointer" href="https://github.com/Kaua-Matheus">Kaua-Matheus</a> </p>
      </div>
    </div>
  )
}
