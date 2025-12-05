import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'

const ThirdPage = () => {
  const [isVisible, setIsVisible] = useState(false)
  const sectionRef = useRef<HTMLElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<{
    scene: THREE.Scene
    camera: THREE.PerspectiveCamera
    renderer: THREE.WebGLRenderer
    cars: THREE.Mesh[]
    animationId: number
  } | null>(null)

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true)
        }
      },
      { threshold: 0.1 }
    )

    if (sectionRef.current) {
      observer.observe(sectionRef.current)
    }

    return () => {
      if (sectionRef.current) {
        observer.unobserve(sectionRef.current)
      }
    }
  }, [])

  useEffect(() => {
    if (!containerRef.current || !isVisible) return

    const roadSpacing = 50
    const carSpeed = 0.1
    const cars: THREE.Mesh[] = []

    // Setup scene
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(75, containerRef.current.clientWidth / containerRef.current.clientHeight, 0.1, 1000)
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight)
    renderer.setClearColor(0x000000, 0)
    containerRef.current.appendChild(renderer.domElement)

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 2)
    scene.add(ambientLight)
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.5)
    directionalLight.position.set(10, 20, 15)
    scene.add(directionalLight)

    // Create building with solar panels
    const createBuilding = (x: number, y: number, z: number, width: number, height: number, depth: number) => {
      const buildingGeometry = new THREE.BoxGeometry(width, height, depth)
      const buildingMaterial = new THREE.MeshLambertMaterial({ color: Math.random() * 0xffffff })
      const building = new THREE.Mesh(buildingGeometry, buildingMaterial)
      building.position.set(x, y + height / 2, z)
      scene.add(building)

      // Solar Panel
      const panelGeometry = new THREE.BoxGeometry(width - 1, 0.2, depth - 1)
      const panelMaterial = new THREE.MeshLambertMaterial({ color: 0x000000 })
      const solarPanel = new THREE.Mesh(panelGeometry, panelMaterial)
      solarPanel.position.set(x, y + height, z)
      scene.add(solarPanel)

      // Grid lines on solar panel
      const lineMaterial = new THREE.LineBasicMaterial({ color: 0xffffff })
      const lineSpacing = 1

      // Horizontal lines
      for (let i = -((depth - 1) / 2); i <= (depth - 1) / 2; i += lineSpacing) {
        const points = [
          new THREE.Vector3(-((width - 1) / 2), 0.1, i),
          new THREE.Vector3(((width - 1) / 2), 0.1, i),
        ]
        const lineGeometry = new THREE.BufferGeometry().setFromPoints(points)
        const line = new THREE.Line(lineGeometry, lineMaterial)
        solarPanel.add(line)
      }

      // Vertical lines
      for (let i = -((width - 1) / 2); i <= (width - 1) / 2; i += lineSpacing) {
        const points = [
          new THREE.Vector3(i, 0.1, -((depth - 1) / 2)),
          new THREE.Vector3(i, 0.1, ((depth - 1) / 2)),
        ]
        const lineGeometry = new THREE.BufferGeometry().setFromPoints(points)
        const line = new THREE.Line(lineGeometry, lineMaterial)
        solarPanel.add(line)
      }

      // Windows
      const windowWidth = 1
      const windowHeight = 1.5
      const windowDepth = 0.1
      const windowMaterial = new THREE.MeshLambertMaterial({ color: 0x87CEFA })

      for (let i = -width / 2 + 1; i < width / 2 - 1; i += 2) {
        for (let j = 1; j < height - 1; j += 2.5) {
          const windowGeometry = new THREE.BoxGeometry(windowWidth, windowHeight, windowDepth)
          const window = new THREE.Mesh(windowGeometry, windowMaterial)
          window.position.set(x + i, y + j + height / 4, z + depth / 2)
          scene.add(window)
        }
      }
    }

    // Create person
    const createPerson = (x: number, z: number) => {
      const bodyGeometry = new THREE.CylinderGeometry(0.2, 0.2, 1.2)
      const bodyMaterial = new THREE.MeshLambertMaterial({ color: Math.random() * 0xffffff })
      const body = new THREE.Mesh(bodyGeometry, bodyMaterial)
      body.position.set(x, 0.6, z)
      scene.add(body)

      const headGeometry = new THREE.SphereGeometry(0.3)
      const headMaterial = new THREE.MeshLambertMaterial({ color: 0xffd700 })
      const head = new THREE.Mesh(headGeometry, headMaterial)
      head.position.set(x, 1.4, z)
      scene.add(head)
    }

    // Create car
    const createCar = (x: number, z: number): THREE.Mesh => {
      const carBodyGeometry = new THREE.BoxGeometry(2, 1, 1)
      const carBodyMaterial = new THREE.MeshLambertMaterial({ color: Math.random() * 0xffffff })
      const carBody = new THREE.Mesh(carBodyGeometry, carBodyMaterial)
      carBody.position.set(x, 0.5, z)
      scene.add(carBody)

      const wheelGeometry = new THREE.CylinderGeometry(0.3, 0.3, 0.5, 32)
      const wheelMaterial = new THREE.MeshLambertMaterial({ color: 0x333333 })

      for (const dx of [-0.8, 0.8]) {
        for (const dz of [-0.5, 0.5]) {
          const wheel = new THREE.Mesh(wheelGeometry, wheelMaterial)
          wheel.rotation.z = Math.PI / 2
          wheel.position.set(x + dx, 0.2, z + dz)
          scene.add(wheel)
        }
      }

      return carBody
    }

    // Create tree
    const createTree = (x: number, z: number) => {
      const trunkGeometry = new THREE.CylinderGeometry(0.2, 0.2, 2)
      const trunkMaterial = new THREE.MeshLambertMaterial({ color: 0x8B4513 })
      const trunk = new THREE.Mesh(trunkGeometry, trunkMaterial)
      trunk.position.set(x, 1, z)
      scene.add(trunk)

      const foliageGeometry = new THREE.SphereGeometry(1)
      const foliageMaterial = new THREE.MeshLambertMaterial({ color: 0x228B22 })
      const foliage = new THREE.Mesh(foliageGeometry, foliageMaterial)
      foliage.position.set(x, 2.5, z)
      scene.add(foliage)
    }

    // Create road
    const createRoad = (x: number, z: number, width: number, depth: number) => {
      const roadGeometry = new THREE.PlaneGeometry(width, depth)
      const roadMaterial = new THREE.MeshLambertMaterial({ color: 0x333333 })
      const road = new THREE.Mesh(roadGeometry, roadMaterial)
      road.rotation.x = -Math.PI / 2
      road.position.set(x, 0.01, z)
      scene.add(road)
    }

    // Create park
    const createPark = (x: number, z: number, width: number, depth: number) => {
      const parkGeometry = new THREE.PlaneGeometry(width, depth)
      const parkMaterial = new THREE.MeshLambertMaterial({ color: 0x32CD32 })
      const park = new THREE.Mesh(parkGeometry, parkMaterial)
      park.rotation.x = -Math.PI / 2
      park.position.set(x, 0.01, z)
      scene.add(park)
    }

    // Generate city
    for (let i = -roadSpacing; i <= roadSpacing; i += 10) {
      for (let j = -roadSpacing; j <= roadSpacing; j += 10) {
        if (Math.random() > 0.3) {
          createBuilding(i, 0, j, 5, Math.random() * 5 + 5, 5)
        } else {
          createPark(i, j, 8, 8)
          createTree(i - 2, j - 2)
          createTree(i + 2, j + 2)
        }
      }
    }

    // Add roads
    for (let i = -roadSpacing; i <= roadSpacing; i += 30) {
      createRoad(i, 0, 50, 5)
      createRoad(0, i, 5, 50)
    }

    // Add people
    for (let i = -roadSpacing; i <= roadSpacing; i += 10) {
      createPerson(i, Math.random() * 10 - 5)
    }

    // Add cars
    for (let i = -roadSpacing; i <= roadSpacing; i += 15) {
      const car = createCar(i, Math.random() * 10 - 5)
      cars.push(car)
    }

    // Position camera
    camera.position.set(50, 50, 50)
    camera.lookAt(0, 0, 0)

    // Animation
    const animate = () => {
      if (sceneRef.current) {
        sceneRef.current.animationId = requestAnimationFrame(animate)
      }
      renderer.render(scene, camera)

      scene.rotation.y += 0.002

      cars.forEach((car) => {
        car.position.x += carSpeed
        if (car.position.x > roadSpacing + 20) {
          car.position.x = -roadSpacing - 20
        }
      })
    }

    animate()

    sceneRef.current = {
      scene,
      camera,
      renderer,
      cars,
      animationId: 0
    }

    // Handle resize
    const handleResize = () => {
      if (!containerRef.current) return
      camera.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight
      camera.updateProjectionMatrix()
      renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight)
    }

    window.addEventListener('resize', handleResize)

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize)
      if (containerRef.current && renderer.domElement) {
        containerRef.current.removeChild(renderer.domElement)
      }
      renderer.dispose()
      scene.clear()
    }
  }, [isVisible])

  return (
    <section 
      ref={sectionRef}
      className={`relative text-white min-h-screen flex items-center justify-center overflow-hidden transition-all duration-1000 ${
        isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-20'
      }`}
    >
      {/* Liquid Glass Overlay */}
      <div className="absolute inset-0 liquid-glass"></div>
      
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 relative z-10 w-full py-8 sm:py-12 md:py-16">
        <div className="max-w-6xl mx-auto w-full">
          {/* Section Title */}
          <div className="text-center mb-8 sm:mb-12 md:mb-16 px-4">
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-extrabold mb-4">
              How It Works
            </h2>
            <p className="text-lg sm:text-xl text-gray-300 max-w-2xl mx-auto">
              Simple three-step process to get your solar analysis
            </p>
          </div>

          {/* 3D City Scene */}
          <div 
            ref={containerRef}
            className="w-full h-[400px] sm:h-[500px] md:h-[600px] rounded-2xl overflow-hidden liquid-glass mx-auto"
            style={{ maxWidth: '100%' }}
          />

          {/* Steps Info */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 sm:gap-8 mt-8 sm:mt-12 px-4">
            <div className="liquid-glass liquid-glass-hover rounded-2xl p-6 sm:p-8 text-center">
              <div className="text-4xl sm:text-5xl mb-4">📸</div>
              <h3 className="text-xl sm:text-2xl font-bold mb-4">01. Upload Images</h3>
              <p className="text-sm sm:text-base text-gray-300 leading-relaxed">
                Upload high-quality photos of your rooftop from multiple angles
              </p>
            </div>

            <div className="liquid-glass liquid-glass-hover rounded-2xl p-6 sm:p-8 text-center">
              <div className="text-4xl sm:text-5xl mb-4">🧠</div>
              <h3 className="text-xl sm:text-2xl font-bold mb-4">02. AI Analysis</h3>
              <p className="text-sm sm:text-base text-gray-300 leading-relaxed">
                Our AI processes your images using YOLO detection and 3D CAD modeling
              </p>
            </div>

            <div className="liquid-glass liquid-glass-hover rounded-2xl p-6 sm:p-8 text-center">
              <div className="text-4xl sm:text-5xl mb-4">📈</div>
              <h3 className="text-xl sm:text-2xl font-bold mb-4">03. Get Results</h3>
              <p className="text-sm sm:text-base text-gray-300 leading-relaxed">
                Receive comprehensive analysis with energy predictions and ROI calculations
              </p>
            </div>
          </div>

          {/* Bottom CTA */}
          <div className="text-center mt-8 sm:mt-12 md:mt-16 px-4">
            <a
              href="#analyze"
              className="inline-flex items-center px-6 sm:px-8 py-3 sm:py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl font-bold text-base sm:text-lg hover:from-blue-700 hover:to-purple-700 transition-all duration-200 shadow-xl hover:shadow-2xl transform hover:scale-105 liquid-glass-hover"
            >
              Start Your Analysis
              <span className="ml-2">→</span>
            </a>
          </div>
        </div>
      </div>
    </section>
  )
}

export default ThirdPage
