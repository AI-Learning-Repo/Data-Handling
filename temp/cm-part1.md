# Coding Marathon 3

The Code Marathon consists of **three sections:**

* **Part A:** API V1, Frontend V1, backend testing, and deployment
* **Part B (Available 13:00):** API V2 with authentication, Frontend V2, testing, and deployment
* **Part C (optional):*** API documentation. 

This document describes **Part A**.

> You don't need to start from scratch! You can use [this starter code.](https://github.com/tx00-resources-en/cm3-starter)


---

# Part A: API V1 & Frontend V1


In Part A, you will build and deploy the first version of the Vehicle Rental application.

This version does **not** use authentication or protected routes.


## 1. Backend: API V1

Implement the following CRUD endpoints for the **VehicleRental** resource.

### GET `/api/vehicleRentals`

Retrieve all vehicle rentals.

### GET `/api/vehicleRentals/:id`

Retrieve a specific vehicle rental.

### POST `/api/vehicleRentals`

Create a new vehicle rental.

### PUT `/api/vehicleRentals/:id`

Update a vehicle rental.

### DELETE `/api/vehicleRentals/:id`

Delete a vehicle rental.

API V1 does **not** require authentication.

---

## 2. Frontend: V1

Build the frontend version that works with **API V1**.

The frontend should allow the user to work with the VehicleRental resource through the API.

This version does **not** include user authentication or protected routes.

---

## 3. Backend Testing

Write backend tests using:

* **Vitest**
* **Supertest**

Tests should cover **all API V1 endpoints**.

Testing should include:

* successful requests
* error handling
* relevant edge cases

---

## 4. Database

API V1 should use its own MongoDB database.

For local development, first test the application using a local MongoDB database.

For deployment, API V1 should use a separate MongoDB Atlas database.

---

## 5. Deployment

Deploy the **complete Part A application** to Render.

The deployed application should include:

* Backend API V1
* Frontend V1
* the corresponding MongoDB database

The deployed application must be working and accessible.


---

# Part A: Completion

Part A is complete when the group has:

* [ ] API V1 CRUD endpoints implemented
* [ ] Backend tests for API V1 implemented
* [ ] Frontend V1 implemented
* [ ] Frontend works with API V1
* [ ] API V1 and Frontend V1 deployed
* [ ] Required OMA deliverables submitted

> A group may continue working on Part A after 12:00 if necessary, but Part A badges are awarded based on completion before the 12:00.
