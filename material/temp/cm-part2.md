# Part B: API V2 & Frontend V2

Part B extends the application from Part A by adding **user administration, authentication, and protected routes**.

You should build Part B based on the working application from Part A.

---

## 1. Backend: API V2

API V2 should contain the same VehicleRental endpoints as API V1.

### VehicleRental endpoints

* **GET** `/api/vehicleRentals`: Retrieve all vehicle rentals
* **GET** `/api/vehicleRentals/:id`: Retrieve a specific vehicle rental
* **POST** `/api/vehicleRentals`: Create a new vehicle rental
* **PUT** `/api/vehicleRentals/:id`: Update a vehicle rental
* **DELETE** `/api/vehicleRentals/:id`: Delete a vehicle rental

### Protected routes

The following operations require authentication:

* **POST** `/api/vehicleRentals`
* **PUT** `/api/vehicleRentals/:id`
* **DELETE** `/api/vehicleRentals/:id`

The read operations remain publicly accessible:

* **GET** `/api/vehicleRentals`
* **GET** `/api/vehicleRentals/:id`

---

## 2. User Administration

Implement the User resource and authentication endpoints.

### POST `/api/auth/signup`

Register a new user.

### POST `/api/auth/login`

Authenticate an existing user.

The application should use authentication tokens to protect the required VehicleRental routes.

### User Model

For this application, we're using **username** instead of email. 


```js
const userSchema = new Schema(
  {
    name: { type: String, required: true },
    username: { type: String, required: true, unique: true },
    password: { type: String, required: true },
    phone_number: { type: String, required: true },  
    licenseNumber: { type: String, required: true, unique: true }, 
    date_of_birth: { type: Date, required: true },   
    address: {
      licenseExpiryDate: { type: Date, required: true },    
      city: { type: String, required: true },
      yearsOfExperience: { type: Number, required: true } 
    }
  },
  { timestamps: true, versionKey: false }
);
```

---

## 3. Backend Testing

Write backend tests using:

* **Vitest**
* **Supertest**

Tests should cover all API V2 endpoints.

Testing should include:

* successful requests
* error handling
* relevant edge cases
* authentication
* unauthenticated access to protected routes
* authenticated access to protected routes

---

## 4. Frontend: V2

Update the frontend to work with API V2.

The frontend should include:

* user registration
* user login
* authentication integration
* access to the VehicleRental functionality
* appropriate handling of protected routes

The frontend should work with the authenticated API V2.

---

## 5. Database

API V2 must use a **separate MongoDB database** from the databases used by the other API versions.

For cloud deployment:

* API V2 → separate loacal database and separate cloud database (i.e MongoDB Atlas)

This separation prevents data conflicts between the different versions.

---

## 6. Deployment

Deploy the **complete Part B application** to Render.

The deployed application should include:

* Backend API V2
* Frontend V2
* the corresponding MongoDB Atlas database

The deployed application must be working and accessible.

---

## 7. Submission

Submit the required deliverables to **OMA before the deadline: 23:45**.

The submitted deliverables must allow the deployed application and the required code to be checked.

---

# Part B: Completion

Part B is complete when the group has:

* [ ] API V2 implemented
* [ ] User administration implemented
* [ ] Signup implemented
* [ ] Login implemented
* [ ] Protected VehicleRental routes implemented
* [ ] Backend tests for API V2 implemented
* [ ] Authentication tests implemented
* [ ] Frontend V2 implemented
* [ ] Registration and login implemented in the frontend
* [ ] Frontend integrated with API V2
* [ ] API V2 and Frontend V2 deployed
* [ ] Separate MongoDB Atlas database used for API V2
* [ ] Required OMA deliverables submitted

**Part B deadline: 16:00**

A group may continue working after 16:00 if necessary. However, Part B badges are awarded based on completion before the stated deadline.
