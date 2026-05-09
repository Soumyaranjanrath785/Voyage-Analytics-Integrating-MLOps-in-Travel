pipeline {
    agent any

    stages {

        stage('Build Docker Image') {
            steps {
                echo 'Building Docker image...'
                sh 'docker build -t mlops-app .'
            }
        }

        stage('Run Container') {
            steps {
                echo 'Running container...'
                sh 'docker run -d -p 8002:8000 mlops-app'
            }
        }

        stage('Test API') {
            steps {
                echo 'Testing API...'
                sh '''
                sleep 5
                curl -X POST http://localhost:8002/predict_price \
                -H "Content-Type: application/json" \
                -d '{"week_day":2,"week_no":10,"day":15}'
                '''
            }
        }
    }
}